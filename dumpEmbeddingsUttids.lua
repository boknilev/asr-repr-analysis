-- TODO snake_case -> camelCase

require 'utils'
require 'UtilsMultiGPU'
require 'lmdb'
require 'nn'
stringx = require 'pl.stringx'
seq = require 'pl.seq'
tds = require 'tds'
 

local cmd = torch.CmdLine()
cmd:option('-embDir', './emb', 'Directory to save embeddings extracted from Deep Speech model, each utterance embeddings in a separate file')
cmd:option('-loadPath', 'deepspeech.t7', 'Path of final model to save/load')
--cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-modelName', 'DeepSpeech', 'Which Deep Speech model is used (DeepSpeech or DeepSpeech-light)')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-lmdbPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb_uttid/train/', 'Path to LMDB dataset')
cmd:option('-trainingSetLMDBPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb_uttid/train/', 'Path to LMDB dataset')
cmd:option('-dictionaryPath', './dictionary', ' File containing the dictionary to use')
cmd:option('-batchSize', 16, 'Batch size')
cmd:option('-reprLayer', 'cnn', 'Deep speech representation to use (cnn | rnn | cnnk (k=1,2) | rnnk (k=1...8) | input')
cmd:option('-convStep', 2, 'Convolution step size in time dimension (2 in deep speech model)')
cmd:option('-removeZeroPadding', true, 'Remove zero padding when dumping embeddings')

local opt = cmd:parse(arg)
print(opt)

  

function init(opt)  
  
  if opt.nGPU > 0 then
    -- needed for loadDataParallel
    require 'cunn'
    require 'cudnn'    
    require 'BatchBRNNReLU'
  end
  print('==> Loading deep speech model')
  local model = loadDataParallel(opt.loadPath, opt.nGPU)
  print(model)
  model:evaluate()
  cnnLayers = model.modules[1]
  if opt.convStep ~= 2 then 
    print('==> Setting CNN step to ' .. opt.convStep .. ' in time timension')
    cnnLayers.modules[1].dW = opt.convStep
    cnnLayers.modules[4].dW = opt.convStep
  end
  rnnLayers = model.modules[2]
  fcLayer = model.modules[3]
  transposeLayer = model.modules[4]
  
  freq = getFreq(opt.trainingSetLMDBPath)    
  
  --[[
  -- first pass: get labels
  print('==> first pass: getting labels')  
  label2idx, idx2label = getLabels(opt.trainingSetLMDBPath)
  local classes = {}
  for idx, _ in ipairs(idx2label) do
    table.insert(classes, idx)
  end
  local num_classes = #idx2label
  print('label2idx:')
  print(label2idx)
  print('idx2label:')
  print(idx2label)
  --print('classes:')
  --print(classes)      
  --]]
  
  collectgarbage()  
  
end

function main(opt)
  
  init(opt)
  
  local dbSpect, dbTrans, dbUttids, dataSize = getSplitDBsUttids(opt.lmdbPath)
  local spects, transcripts, uttids = loadDataUttids(dbSpect, dbTrans, dbUttids)
  
  print('Getting embeddings')
  embeddings = extractEmbeddings(spects, transcripts, uttids, dataSize, opt)
  assert(#embeddings == #uttids, 'incompatible number of utterance embeddings and utterances')
  print('Writing embeddings per utterance')
  local c = 0
  for uttEmbeddings, uttid in seq.zip(embeddings, uttids) do
    c = c + 1
    xlua.progress(c, #uttids)
    writeMatrixToFile(uttEmbeddings, paths.concat(opt.embDir, uttid .. '.emb'))
  end  
  print('done')
  
  collectgarbage()
  
end  


function extractEmbeddings(allSpects, allTranscripts, allUttids, dataSize, opt)
  local time = sys.clock()
  local shuffle = torch.range(1, dataSize) -- no need to shuffle on test TODO: clean this
  
  -- determine embedding size
  local embSize
  if opt.reprLayer == 'cnn' or opt.reprLayer == 'cnn2' then
    embSize = 32*41 
  elseif opt.reprLayer == 'cnn1' then
    embSize = 32*61
  elseif stringx.startswith(opt.reprLayer, 'rnn') then
    if opt.modelName == 'DeepSpeech' then
      embSize = 1760 
    elseif opt.modelName == 'DeepSpeech-light' then
      embSize = 600
    else
      error('unsupported modelName ' .. opt.modelName)
    end
  elseif opt.reprLayer == 'input' then
    embSize = freq
  else
    error('unsuppoerted reprLayer ' .. opt.reprLayer)
  end  
  print('Embedding size: ' .. embSize)
  
  local input = torch.Tensor()
  if opt.nGPU > 0 then
    input = input:cuda()
  end
  
  -- container for embeddings and labels
  local embeddings = {}
  
  for i = 1,dataSize,opt.batchSize do
    collectgarbage()
    xlua.progress(i, dataSize)
    
    -- get next batch
    local indices = shuffle[{ {i, math.min(i+opt.batchSize-1, dataSize) } }]
    --local inputsCPU, targets, sizes, transcripts, times = nextBatch(indices, dbSpect, dbTrans, dbTimes)
    --local inputsCPU, _, transcripts, times = nextBatchOld(indices, dbSpect, dbTrans, dbTimes)
    local inputsCPU, _, transcripts, uttids = nextBatchUttids(indices, allSpects, allTranscripts, allUttids)
    input:resize(inputsCPU:size()):copy(inputsCPU) -- batch size X 1 X freq X input seq length
              
    local repr, batchDim, timeDim
    if opt.reprLayer == 'cnn' or opt.reprLayer == 'cnn2' then
      repr = cnnLayers:forward(input)
      batchDim, timeDim = 2, 1
    elseif opt.reprLayer == 'cnn1' then 
      repr = cnnLayers.modules[3]:forward(cnnLayers.modules[2]:forward(cnnLayers.modules[1]:forward(input)))
      -- TODO define this somewhere else and don't hard code sizes
      local reshapeRepr = nn.Sequential()
      reshapeRepr:add(nn.View(32*61, -1):setNumInputDims(3)) -- batch size X 32*61 X convolved seq length
      reshapeRepr:add(nn.Transpose({ 2, 3 }, { 1, 2 })) -- convolved seq length X batch size X 32*61
      if opt.nGPU > 0 then reshapeRepr = reshapeRepr:cuda() end
      repr = reshapeRepr:forward(repr)
      batchDim, timeDim = 2, 1      
    elseif stringx.startswith(opt.reprLayer, 'rnn') then 
      repr = cnnLayers:forward(input)
      if opt.reprLayer == 'rnn' or opt.reprLayer == 'rnn7' then          
        repr = rnnLayers:forward(repr) -- convolved seq length X batch size X 1760
      else
        local rnnLayerNum = tonumber(opt.reprLayer:sub(opt.reprLayer:len()))
        assert(rnnLayerNum and rnnLayerNum > 0 and rnnLayerNum < 7, 'bad reprLayer ' .. opt.reprLayer .. '\n')
        
        repr = rnnLayers.modules[1]:forward(repr) -- first rnn layer
        for i = 1,rnnLayerNum - 1 do
          repr = rnnLayers.modules[i*2]:forward(repr) -- batch norm layer
          repr = rnnLayers.modules[i*2+1]:forward(repr) -- next rnn layer
        end
      end
      batchDim, timeDim = 2, 1
    elseif opt.reprLayer == 'input' then
      repr = input:squeeze():transpose(2,3):transpose(1,2)
      batchDim, timeDim = 2, 1
    else
      error('unsupported representation ' .. opt.reprLayer)
    end
      
    -- iterate over batch (TODO: can vectorize this?)     
    for k = 1, repr:size(batchDim) do 
      local goldLabels = {}, {}
      -- iterate over time
      local uttEmbeddings = {}
      for t = 1, repr:size(timeDim) do 
        local embedding = repr[t][k]:reshape(1, embSize):double()        
        if (not opt.removeZeroPadding) or (not torch.all(embedding:eq(torch.zeros(embedding:size())))) then
          table.insert(uttEmbeddings, embedding)
        end
      end      
      uttEmbeddings = nn.JoinTable(1):forward(uttEmbeddings)
      table.insert(embeddings, uttEmbeddings)
    end        
  
  end
    
  time = (sys.clock() - time) / dataSize
  print('==> time to extract 1 sample = ' .. (time*1000) .. 'ms') 
  
  collectgarbage()

  return embeddings
      
end

main(opt)