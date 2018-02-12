-- TODO snake_case -> camelCase

require 'utils'
require 'UtilsMultiGPU'
require 'lmdb'
require 'nn'
stringx = require 'pl.stringx'
tds = require 'tds'


local cmd = torch.CmdLine()
cmd:option('-embFile', 'emb.t7', 'File to save embeddings extracted from Deep Speech model')
cmd:option('-labelFile', 'lbl.t7', 'File to save gold labels corresponding to each embedding')
cmd:option('-loadPath', 'deepspeech.t7', 'Path of final model to save/load')
--cmd:option('-modelName', 'DeepSpeechModel', 'Name of class containing architecture')
cmd:option('-modelName', 'DeepSpeech', 'Which Deep Speech model is used (DeepSpeech or DeepSpeech-light)')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-lmdbPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb/train/', 'Path to LMDB dataset')
cmd:option('-trainingSetLMDBPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb/train/', 'Path to LMDB dataset')
cmd:option('-dictionaryPath', './dictionary', ' File containing the dictionary to use')
cmd:option('-batchSize', 16, 'Batch size')
cmd:option('-reprLayer', 'cnn', 'Deep speech representation to use (cnn | rnn | cnnk (k=1,2) | rnnk (k=1...8) | input')
cmd:option('-convStep', 2, 'Convolution step size in time dimension (2 in deep speech model)')

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
  
  collectgarbage()  
  
end

function main(opt)
  
  init(opt)
  
  local dbSpect, dbTrans, dbTimes, dataSize = getSplitDBs(opt.lmdbPath)
  local spects, transcripts, times = loadData(dbSpect, dbTrans, dbTimes)
  
  print('Getting embeddings and labels')
  embeddings, labels = extractEmbeddings(spects, transcripts, times, dataSize, opt)
  print('Dumping embeddings to file: ' .. opt.embFile)
  torch.save(opt.embFile, embeddings)
  print('Dumping labels to file: ' .. opt.labelFile)
  torch.save(opt.labelFile, labels)
  
  collectgarbage()
  
end  



function extractEmbeddings(allSpects, allTranscripts, allTimes, dataSize, opt)
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
  local embeddings, labels = {}, {}
  
  local num_total_phonemes = 0
  for i = 1,dataSize,opt.batchSize do
    collectgarbage()
    xlua.progress(i, dataSize)
    
    -- get next batch
    local indices = shuffle[{ {i, math.min(i+opt.batchSize-1, dataSize) } }]
    --local inputsCPU, targets, sizes, transcripts, times = nextBatch(indices, dbSpect, dbTrans, dbTimes)
    --local inputsCPU, _, transcripts, times = nextBatchOld(indices, dbSpect, dbTrans, dbTimes)
    local inputsCPU, _, transcripts, times = nextBatch(indices, allSpects, allTranscripts, allTimes)
    input:resize(inputsCPU:size()):copy(inputsCPU) -- batch size X 1 X freq X input seq length
          
    local num_phonemes = 0, 0    
    
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
      for t = 1, repr:size(timeDim) do 
        -- get label id corresponding to current frame; if frame is outside of transcript (because of padding), will return 0
        local label = getFrameLabel(transcripts[k], times[k], t, opt.reprLayer, opt.convStep)
        if label then
          --print(repr[t][k])
          --print(repr[t][k]:reshape(embSize, 1))
          -- view doesn't work here
          local embedding = repr[t][k]:reshape(embSize, 1):double()
          table.insert(embeddings, embedding)
          table.insert(labels, label)
          num_phonemes = num_phonemes + 1          
        end            
      end      
    end        
  
    num_total_phonemes = num_total_phonemes + num_phonemes
  
  end
  
  local embMat = nn.JoinTable(2):forward(embeddings)
  
  time = (sys.clock() - time) / dataSize
  print('==> time to extract 1 sample = ' .. (time*1000) .. 'ms') 
  print('==> total number of phonemes (frames): ' .. num_total_phonemes)
  print('==> size of embedding matrix:')
  print(embMat:size())
  
  collectgarbage()

  return embMat, labels
      
end

main(opt)