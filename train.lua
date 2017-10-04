-- uses code from:
-- https://github.com/SeanNaren/deepspeech.torch

require 'utils'
require 'UtilsMultiGPU'
require 'lmdb'
require 'optim'
require 'nn'
stringx = require 'pl.stringx'


local cmd = torch.CmdLine()
cmd:option('-saveClassifierModel', false, 'Save model after training/testing')
cmd:option('-saveClassifierPath', false, 'Path to save classifier model')
cmd:option('-savePredDir', '.', 'Directory to save predictions')
cmd:option('-predFile', 'pred.txt', 'File to save classifier predictions on test set')
cmd:option('-trainPredFile', '', 'File to save classifier predictions on train set (leave empty for not saving predictions on train)')
cmd:option('-loadPath', 'deepspeech.t7', 'Path of final model to save/load')
cmd:option('-modelName', 'DeepSpeech', 'Which Deep Speech model is used (DeepSpeech or DeepSpeech-light)')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-trainingSetLMDBPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb/train/', 'Path to LMDB training dataset')
cmd:option('-validationSetLMDBPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb/test/', 'Path to LMDB test dataset')
cmd:option('-testSetLMDBPath', '/data/sls/scratch/belinkov/asr/prediction/data/timit_lmdb/test/', 'Path to LMDB test dataset')
--cmd:option('-logsTrainPath', './logs/TrainingLoss/', ' Path to save Training logs')
--cmd:option('-logsValidationPath', './logs/ValidationScores/', ' Path to save Validation logs')
cmd:option('-logFile', './log.txt', 'File to save logs')
cmd:option('-plot', false, 'Plot loss and accuracy')
cmd:option('-dictionaryPath', 'dictionary', ' File containing the dictionary to use')
cmd:option('-batchSize', 16, 'Batch size in training')
cmd:option('-validationBatchSize', 1, 'Batch size for validation')
cmd:option('-patience', 5, 'Patience when training the classifier')
cmd:option('-optim', 'ADAM', 'Optimizer to use in the classifier (ADAM/ADAGRAD/ADADELTA/SGD)')
cmd:option('-classifierSize', 500, 'Classifier hidden layer size')
cmd:option('-linearClassifier', false, 'Use linear classifier')
cmd:option('-epochs', 30, 'Number of epochs for training the classifier')
cmd:option('-learningRate', 0.001, 'Learning rate for the classifier')
cmd:option('-reprLayer', 'cnn', 'Deep speech representation to use (cnn | rnn | cnnk (k=1,2) | rnnk (k=1...8) | input')
cmd:option('-convStep', 2, 'Convolution step size in time dimension (2 in deep speech model)')
cmd:option('-phoneClasses', false, 'Use phone classes (must also specify phoneClassesFile)')
cmd:option('-phoneClassesFile', '', 'File containing list of phone classes, every line has: phone<SPACE>class')
cmd:option('-aggregateFrames', false, 'Aggregrate frames in each phone (by default as an average) and do phonme classification')
cmd:option('-writeCTCPredictions', false, 'Write predictions of the full CTC model to file')
cmd:option('-window', 0, 'Number of frames on each side of the current frame to use as features for classification')

local opt = cmd:parse(arg)
print(opt)

  

function init(opt)  
  
  opt.predFile = paths.concat(opt.savePredDir, opt.predFile)  
  if opt.trainPredFile:len() > 0 then 
    opt.trainPredFile = paths.concat(opt.savePredDir, opt.trainPredFile)
  else
    opt.trainPredFile = nil
  end
  
  if opt.nGPU > 0 then
    -- needed for loadDataParallel
    require 'cunn'
    require 'cudnn'    
    require 'BatchBRNNReLU'
  end
  print('==> Loading deep speech model')
  model = loadDataParallel(opt.loadPath, opt.nGPU)
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
  
  --local modelDef = require(opt.modelName)
  --calSize = modelDef[2]
  
  local freq = getFreq(opt.trainingSetLMDBPath)    
  
  -- first pass: get labels
  print('==> first pass: getting labels')  
  label2idx, idx2label = getLabels(opt.trainingSetLMDBPath)
  local classes = {}
  for idx, _ in ipairs(idx2label) do
    table.insert(classes, idx)
  end
  print('label2idx:')
  print(label2idx)
  print('idx2label:')
  print(idx2label)
  print('classes:')
  print(classes)      
  
  local numClasses = #idx2label
  
  if opt.phoneClasses then
    assert(paths.filep(opt.phoneClassesFile), 'bad file in opt.phoneClassesFile')
    print('==> loading phone classes')
    phone2class, phoneClasses = getPhoneClasses(opt.phoneClassesFile)
    numClasses = #phoneClasses
    print('==> replacing labels with phone classes')
    label2idx, idx2label = {}, {}
    for i, label in pairs(phoneClasses) do
      if not label2idx[label] then
        idx2label[#idx2label+1] = label
        label2idx[label] = #idx2label         
      end
    end
    classes = {}
    for idx, _ in ipairs(idx2label) do
      table.insert(classes, idx)
    end
    print('label2idx:')
    print(label2idx)
    print('idx2label:')
    print(idx2label)
    print('classes:')
    print(classes)    
    numClasses = #idx2label
  end
  
  
  -- define classifier
  local classifierInputSize
  if opt.reprLayer == 'cnn' or opt.reprLayer == 'cnn2' then
    classifierInputSize = 32*41 
  elseif opt.reprLayer == 'cnn1' then
    classifierInputSize = 32*61
  elseif stringx.startswith(opt.reprLayer, 'rnn') then
    if opt.modelName == 'DeepSpeech' then
      classifierInputSize = 1760 
    elseif opt.modelName == 'DeepSpeech-light' then
      classifierInputSize = 600
    else
      error('unsupported modelName ' .. opt.modelName)
    end
  elseif opt.reprLayer == 'input' then
    classifierInputSize = freq
  else
    error('unsuppoerted reprLayer ' .. opt.reprLayer)
  end
  
  if opt.window > 0 then
    print('==> using window of ' .. opt.window .. ' frames around current frame')
    classifierInputSize = (2 * opt.window + 1) * classifierInputSize
  end
  
  classifier = nn.Sequential()
  if opt.linearClassifier then
    classifier:add(nn.Linear(classifierInputSize, numClasses))
  else
    classifier:add(nn.Linear(classifierInputSize, opt.classifierSize))
    classifier:add(nn.Dropout(opt.classifierDropout))
    classifier:add(nn.ReLU(true))
    classifier:add(nn.Linear(opt.classifierSize, numClasses)) 
  end
  
  print('==> defined classification model:')
  print(classifier)
    
  -- define classification criterion
  criterion = nn.CrossEntropyCriterion()  
  
  -- move to cuda
  if opt.nGPU > 0 then     
    classifier = classifier:cuda()
    criterion = criterion:cuda()
  end
  
  -- get classifier parameters and gradients
  classifierParams, classifierGrads = classifier:getParameters()
  
  -- define optimizer
  if opt.optim == 'ADAM' then
    optimState = {learningRate = opt.learningRate}
    optimMethod = optim.adam
  elseif opt.optim == 'ADAGRAD' then
    optimState = {learningRate = opt.learningRate}
    optimMethod = optim.adagrad
  elseif opt.optim == 'ADADELTA' then
    optimState = {}
    optimMethod = optim.adadelta
  else
    optimState = {learningRate = opt.learningRate}
    optimMethod = optim.sgd
  end  
  
  confusion = optim.ConfusionMatrix(classes)  
  
  logger = optim.Logger(opt.logFile)
  logger:setNames{'Train loss', 'Train accuracy', 'Validation loss', 'Validation accuracy', 'Test loss', 'Test accuracy'}
  logger:style{'+-', '+', '+-', '+', '+-', '+'}
  
  if opt.writeCTCPredictions then
    require 'Mapper'
    mapper = Mapper(opt.dictionaryPath)
  end
  
  
  collectgarbage()  
  
end

function main(opt)
  
  init(opt)
  
  local dbSpectTrain, dbTransTrain, dbTimesTrain, trainDataSize = getSplitDBs(opt.trainingSetLMDBPath)
  local dbSpectVal, dbTransVal, dbTimesVal, valDataSize = getSplitDBs(opt.validationSetLMDBPath)
  local dbSpectTest, dbTransTest, dbTimesTest, testDataSize = getSplitDBs(opt.testSetLMDBPath)
  
  local trainSpects, trainTranscripts, trainTimes = loadData(dbSpectTrain, dbTransTrain, dbTimesTrain)
  local valSpects, valTranscripts, valTimes = loadData(dbSpectVal, dbTransVal, dbTimesVal)
  local testSpects, testTranscripts, testTimes = loadData(dbSpectTest, dbTransTest, dbTimesTest)  
  
  -- do epochs
  local epoch, bestEpoch, bestLoss = 1, 1, math.huge
  while epoch <= opt.epochs and epoch - bestEpoch <= opt.patience do 
    trainLoss, trainAcc = train(trainSpects, trainTranscripts, trainTimes, trainDataSize, epoch, opt, opt.trainPredFile)
    valLoss, valAcc = eval(valSpects, valTranscripts, valTimes, valDataSize, epoch, opt, 'val')
    if valLoss < bestLoss then
      bestEpoch = epoch
      bestLoss = valLoss
      if opt.saveClassifiermodel == 1 then
        -- save current model
        local filename = paths.concat(opt.saveClassifierPath, 'classifier_model_epoch_' .. epoch .. '.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, classifier)        
      end
    end
    testLoss, testAcc = eval(testSpects, testTranscripts, testTimes, testDataSize, epoch, opt, 'test', opt.predFile)
    print('finished epoch ' .. epoch .. ', with val loss: ' .. valLoss)
    print('best epoch: ' .. bestEpoch .. ', with val loss: ' .. bestLoss)
    logger:add{trainLoss, trainAcc, valLoss, valAcc, testLoss, testAcc}
    if opt.plot then logger:plot() end
    
    epoch = epoch + 1    
    collectgarbage(); collectgarbage();
  end
  if epoch - bestEpoch > opt.patience then
    print('==> reached patience of ' .. opt.patience .. ' epochs, stopping...')
  end    
end  



function train(allSpects, allTranscripts, allTimes, dataSize, epoch, opt, predFilename)
  local time = sys.clock()
  classifier:training()
  
  local predFile, goldFile, transFile
  if predFilename then
    predFile = torch.DiskFile(predFilename .. '.epoch' .. epoch, 'w')
    goldFile = torch.DiskFile(predFilename .. '.gold', 'w')
    transFile = torch.DiskFile(predFilename .. '.trans', 'w')
  end  
  
  local shuffle = torch.randperm(dataSize)
  
  local input = torch.Tensor()
  if opt.nGPU > 0 then
    input = input:cuda()
  end
  
  
  print('\n==> doing epoch on training data:')
  print('\n==> epoch # ' .. epoch .. ' [batch size = ' .. opt.batchSize .. ']')  
  
  local totalLoss, numTotalPhonemes = 0, 0
  for i = 1,dataSize,opt.batchSize do    
    collectgarbage()
    xlua.progress(i, dataSize)
    
    -- get next batch
    local indices = shuffle[{ {i, math.min(i+opt.batchSize-1, dataSize) } }]
    local inputsCPU, _, transcripts, times = nextBatch(indices, allSpects, allTranscripts, allTimes)
    input:resize(inputsCPU:size()):copy(inputsCPU) -- batch size X 1 X freq X input seq length
    
    -- closure
    local evalLossGrad = function(x) 
      -- get new params
      if x ~= classifierParams then classifierParams:copy(x) end
      
      -- reset gradients
      classifierGrads:zero()
      
      local loss, numPhonemes = 0, 0    
            
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
        repr = input:squeeze():transpose(2,3):transpose(1,2) -- convolved seq length X batch size X freq
        batchDim, timeDim = 2, 1
      else
        error('unsupported representation ' .. opt.reprLayer)
      end
      
      
      -- iterate over batch dim
      for k = 1, repr:size(batchDim) do 
        local predLabels, goldLabels = {}, {}
        -- iterate over time 
        for t = 1, repr:size(timeDim) do 
          -- get label id corresponding to current frame; if frame is outside of transcript (because of padding), will return 0
          local label = getFrameLabel(transcripts[k], times[k], t, opt.reprLayer, opt.convStep)
          if opt.phoneClasses then
            label = phone2class[label]
          end
          if label then
            local target = label2idx[label]
            local classifierInput = repr[t][k]
            if opt.window > 0 then
              classifierInput = getWindowedInput(opt.window, repr, t, k, timeDim)
            end
            local classifierOut = classifier:forward(classifierInput)
            loss = loss + criterion:forward(classifierOut, target)
            numPhonemes = numPhonemes + 1
            local outputGrad = criterion:backward(classifierOut, target)
            classifier:backward(classifierInput, outputGrad)
                        
            -- get predicted labels to write to file
            if predFile then
              local _, predIdx =  classifierOut:max(1)
              predIdx = predIdx:long()[1]
              local predLabel = idx2label[predIdx]
              table.insert(predLabels, predLabel)
              table.insert(goldLabels, label)
            end               
            
            -- update confusion matrix
            confusion:add(classifierOut, target)
          end            
        end
        if predFile then
          predFile:writeString(stringx.join(' ', predLabels) .. '\n')
          goldFile:writeString(stringx.join(' ', goldLabels) .. '\n')
          transFile:writeString(transcripts[k] .. '\n')
        end        
      end        
    
      classifierGrads:div(numPhonemes)
      -- keep loss over entire training data
      totalLoss = totalLoss + loss
      numTotalPhonemes = numTotalPhonemes + numPhonemes
      -- loss for current batch
      loss = loss/numPhonemes
      
      return loss, classifierGrads      
    end
    
    optimMethod(evalLossGrad, classifierParams, optimState)
  
  end
  
  time = (sys.clock() - time) / dataSize
  print('==> time to learn 1 sample = ' .. (time*1000) .. 'ms') 
  totalLoss = totalLoss/numTotalPhonemes
  print('==> loss: ' .. totalLoss)  
  print(confusion)
  print('==> total number of train phonemes (frames): ' .. numTotalPhonemes)
  local accuracy = confusion.totalValid * 100
     
  -- for next epoch
  confusion:zero()
  
  if predFile then predFile:close() end
  if goldFile then goldFile:close() end
  if transFile then transFile:close() end
  
  return totalLoss, accuracy
      
end


function eval(allSpects, allTranscripts, allTimes, dataSize, epoch, opt, testOrVal, predFilename)
  local testOrVal = testOrVal or 'test'  
  local predFile, goldFile, transFile, ctcPredFile
  if predFilename then
    predFile = torch.DiskFile(predFilename .. '.epoch' .. epoch, 'w')
    goldFile = torch.DiskFile(predFilename .. '.gold', 'w')
    transFile = torch.DiskFile(predFilename .. '.trans', 'w')
    if opt.writeCTCPredictions then
      ctcPredFile = torch.DiskFile(predFilename .. '.ctc.pred', 'w')
    end      
  end  
  local time = sys.clock()
  classifier:evaluate()
  local shuffle = torch.range(1, dataSize) -- no need to shuffle on test TODO: clean this
  
  local input = torch.Tensor()
  if opt.nGPU > 0 then
    input = input:cuda()
  end
  
  print('\n==> evaluating on ' .. testOrVal .. ' data')
  print('\n==> epoch # ' .. epoch .. ' [batch size = ' .. opt.batchSize .. ']')  
  
  local totalLoss, numTotalPhonemes = 0, 0
  for i = 1,dataSize,opt.batchSize do
    collectgarbage()
    xlua.progress(i, dataSize)
    
    -- get next batch
    local indices = shuffle[{ {i, math.min(i+opt.batchSize-1, dataSize) } }]
    local inputsCPU, _, transcripts, times = nextBatch(indices, allSpects, allTranscripts, allTimes)
    input:resize(inputsCPU:size()):copy(inputsCPU) -- batch size X 1 X freq X input seq length
          
    local loss, numPhonemes = 0, 0    
    
    -- to compute the convolved seq length = ( ((input seq length - 11) / 2 + 1) - 11 ) / 2 + 1, or use the function calSize(input seq length)
    
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
    
    -- write CTC predictions (including blanks)
    local ctcBatchPredictions
    if opt.writeCTCPredictions and ctcPredFile then -- and opt.convStep == 1 then
      ctcBatchPredictions = model:forward(input)
    end
    --[[
      local size = ctcPredictions:size(1)
      for j = 1, size do
        local prediction = ctcPredictions[j]
        local _, maxIndices = torch.max(prediction, 2)
        maxIndices = maxIndices:float():squeeze()
        local tokens = {}
        for i=1, maxIndices:size(1) do
          local token = maxIndices[i] - 1
          table.insert(tokens, token)
        end
        --local predict_tokens = mapper:decodeOutput(prediction)
        local predictTranscript = mapper:tokensToText(tokens)
        ctcPredFile:writeString(predictTranscript .. '\n')
      end
    end
    --]]
      
    -- iterate over batch (TODO: vectorize this?) 
    for k = 1, repr:size(batchDim) do 
      
      -- ctc predictions for current uttterance 
      local ctcMaxIndices
      if opt.writeCTCPredictions and ctcPredFile and ctcBatchPredictions then --and opt.convStep == 1 then
        _, ctcMaxIndices = torch.max(ctcBatchPredictions[k], 2)
        ctcMaxIndices = ctcMaxIndices:float():squeeze()
      end
        
      local predLabels, goldLabels, predCTCTokens = {}, {}, {}
      -- iterate over time
      for t = 1, repr:size(timeDim) do 
        -- get label id corresponding to current frame; if frame is outside of transcript (because of padding), will return 0
        local label = getFrameLabel(transcripts[k], times[k], t, opt.reprLayer, opt.convStep)
        if opt.phoneClasses then
          label = phone2class[label]
        end        
        if label then
          local target = label2idx[label]
          local classifierInput = repr[t][k]
          if opt.window > 0 then
            classifierInput = getWindowedInput(opt.window, repr, t, k, timeDim)
          end          
          
          local classifierOut = classifier:forward(classifierInput)
          loss = loss + criterion:forward(classifierOut, target)
          numPhonemes = numPhonemes + 1
          
          -- get predicted labels to write to file
          if predFile then
            local _, predIdx =  classifierOut:max(1)
            predIdx = predIdx:long()[1]
            local predLabel = idx2label[predIdx]
            table.insert(predLabels, predLabel)
            table.insert(goldLabels, label)
            
            if opt.writeCTCPredictions and ctcPredFile and ctcMaxIndices then --and opt.convStep == 1 then
              local token = ctcMaxIndices[t] - 1
              table.insert(predCTCTokens, token)
            end
          end          
          
          -- update confusion matrix
          confusion:add(classifierOut, target)
        end            
      end
      if predFile then
        predFile:writeString(stringx.join(' ', predLabels) .. '\n')
        goldFile:writeString(stringx.join(' ', goldLabels) .. '\n')
        transFile:writeString(transcripts[k] .. '\n')
        if ctcPredFile then
          local predictTranscript = mapper:tokensToText(predCTCTokens)
          --ctcPredFile:writeString(predictTranscript .. '\n')
          ctcPredFile:writeString(stringx.join(' ', predCTCTokens) .. '\n')
        end
      end      
    end        
  
    classifierGrads:div(numPhonemes)
    -- keep loss over entire training data
    totalLoss = totalLoss + loss
    numTotalPhonemes = numTotalPhonemes + numPhonemes
    -- loss for current batch
    loss = loss/numPhonemes
      
    
  end
  
  time = (sys.clock() - time) / dataSize
  print('==> time to evaluate 1 sample = ' .. (time*1000) .. 'ms') 
  totalLoss = totalLoss/numTotalPhonemes
  print('==> loss: ' .. totalLoss)  
  print(confusion)
  print('==> total number of ' .. testOrVal .. ' phonemes (frames): ' .. numTotalPhonemes)
  local accuracy = confusion.totalValid * 100
     
  -- for next epoch
  confusion:zero()
  
  if predFile then predFile:close() end
  if goldFile then goldFile:close() end
  if transFile then transFile:close() end
  if ctcPredFile then ctcPredFile:close() end
  
  return totalLoss, accuracy  
      
end
  
  
function trainAggregateFrames(allSpects, allTranscripts, allTimes, dataSize, epoch, opt, predFilename)
  local time = sys.clock()
  classifier:training()
  
  local predFile, goldFile, transFile
  if predFilename then
    predFile = torch.DiskFile(predFilename .. '.epoch' .. epoch, 'w')
    goldFile = torch.DiskFile(predFilename .. '.gold', 'w')
    transFile = torch.DiskFile(predFilename .. '.trans', 'w')
  end  
  
  local shuffle = torch.randperm(dataSize)
  
  local input = torch.Tensor()
  if opt.nGPU > 0 then
    input = input:cuda()
  end
  
  
  print('\n==> doing epoch on training data:')
  print('\n==> epoch # ' .. epoch .. ' [batch size = ' .. opt.batchSize .. ']')  
  
  local totalLoss, numTotalPhonemes = 0, 0
  --for i = 1,100,opt.batchSize do
  for i = 1,dataSize,opt.batchSize do    
    collectgarbage()
    xlua.progress(i, dataSize)
    
    -- get next batch
    local indices = shuffle[{ {i, math.min(i+opt.batchSize-1, dataSize) } }]
    local inputsCPU, _, transcripts, times = nextBatch(indices, allSpects, allTranscripts, allTimes)
    input:resize(inputsCPU:size()):copy(inputsCPU) -- batch size X 1 X freq X input seq length
    
    -- closure
    local evalLossGrad = function(x) 
      -- get new params
      if x ~= classifierParams then classifierParams:copy(x) end
      
      -- reset gradients
      classifierGrads:zero()
      
      local loss, numPhonemes = 0, 0    
            
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
        repr = input:squeeze():transpose(2,3):transpose(1,2) -- convolved seq length X batch size X freq
        batchDim, timeDim = 2, 1
      else
        error('unsupported representation ' .. opt.reprLayer)
      end
      
      
      -- iterate over batch dim
      for k = 1, repr:size(batchDim) do 
        local predLabels, goldLabels = {}, {}
        local prevTarget, curNumFrames, aggregateRepr = 0, 0, torch.zeros(repr[1][k]:nElement())
        if opt.nGPU > 0 then aggregateRepr = aggregateRepr:cuda() end
        -- iterate over time 
        for t = 1, repr:size(timeDim) do 
          -- get label id corresponding to current frame; if frame is outside of transcript (because of padding), will return 0
          local label = getFrameLabel(transcripts[k], times[k], t, opt.reprLayer, opt.convStep)
          --print('label:')
          --print(label)
          if opt.phoneClasses then
            label = phone2class[label]
          end
          if label then
            local target = label2idx[label]
            
            -- aggregate frames
            if opt.aggregateFrames then
              if prevTarget > 0 then
                if prevTarget == target then
                  -- aggregate frames
                  curNumFrames = curNumFrames + 1
                  aggregateRepr:add(repr[t][k])
                else
                  -- found new phoneme; make a forward/backward step on previous one
                  -- input to classifier is average over frames
                  local classifierInput = aggregateRepr:div(curNumFrames)
                  local classifierOut = classifier:forward(classifierInput)
                  loss = loss + criterion:forward(classifierOut, target)
                  numPhonemes = numPhonemes + 1
                  local outputGrad = criterion:backward(classifierOut, target)
                  classifier:backward(classifierInput, outputGrad)
                  -- get predicted labels to write to file
                  if predFile then
                    local _, predIdx =  classifierOut:max(1)
                    predIdx = predIdx:long()[1]
                    local predLabel = idx2label[predIdx]
                    table.insert(predLabels, predLabel)
                    table.insert(goldLabels, label)
                  end               
                  
                  -- update confusion matrix
                  confusion:add(classifierOut, target)
                  
                  -- init for next phoneme
                  curNumFrames = 1
                  aggregateRepr:copy(repr[t][k])
                end
              else
                -- first frame in the utterance
                curNumFrames = 1
                aggregateRepr:copy(repr[t][k])
              end
              -- update previous target
              prevTarget = target
              
              -- clean last frame
              if t == repr:size(timeDim) and curNumFrames > 0 then
                local classifierInput = aggregateRepr:div(curNumFrames)
                local classifierOut = classifier:forward(classifierInput)
                loss = loss + criterion:forward(classifierOut, target)
                numPhonemes = numPhonemes + 1
                local outputGrad = criterion:backward(classifierOut, target)
                classifier:backward(classifierInput, outputGrad)
                -- get predicted labels to write to file
                if predFile then
                  local _, predIdx =  classifierOut:max(1)
                  predIdx = predIdx:long()[1]
                  local predLabel = idx2label[predIdx]
                  table.insert(predLabels, predLabel)
                  table.insert(goldLabels, label)
                end               
                
                -- update confusion matrix
                confusion:add(classifierOut, target)
              end
                
            else
            -- classify each frame, without aggregating            
            
              local classifierInput = repr[t][k]
              local classifierOut = classifier:forward(classifierInput)
              loss = loss + criterion:forward(classifierOut, target)
              numPhonemes = numPhonemes + 1
              local outputGrad = criterion:backward(classifierOut, target)
              classifier:backward(classifierInput, outputGrad)
                          
              -- get predicted labels to write to file
              if predFile then
                local _, predIdx =  classifierOut:max(1)
                predIdx = predIdx:long()[1]
                local predLabel = idx2label[predIdx]
                table.insert(predLabels, predLabel)
                table.insert(goldLabels, label)
              end               
              
              -- update confusion matrix
              confusion:add(classifierOut, target)
            end
            
          elseif opt.aggregateFrames and curNumFrames > 0 then
            -- if found invalid label (e.g. silence), and aggregating frames, and have some frames aggregated, make a forward/backword step
            local target = prevTarget
            local classifierInput = aggregateRepr:div(curNumFrames)
            local classifierOut = classifier:forward(classifierInput)
            loss = loss + criterion:forward(classifierOut, target)
            numPhonemes = numPhonemes + 1
            local outputGrad = criterion:backward(classifierOut, target)
            classifier:backward(classifierInput, outputGrad)
            -- get predicted labels to write to file
            if predFile then
              local _, predIdx =  classifierOut:max(1)
              predIdx = predIdx:long()[1]
              local predLabel = idx2label[predIdx]
              table.insert(predLabels, predLabel)
              table.insert(goldLabels, label)
            end               
            
            -- update confusion matrix
            confusion:add(classifierOut, target) 
            
            -- init for next phoneme
            prevTarget, curNumFrames, aggregateRepr = 0, 0, torch.zeros(repr[1][k]:nElement())
            if opt.nGPU > 0 then aggregateRepr = aggregateRepr:cuda() end
          end            
        end
        if predFile then
          predFile:writeString(stringx.join(' ', predLabels) .. '\n')
          goldFile:writeString(stringx.join(' ', goldLabels) .. '\n')
          transFile:writeString(transcripts[k] .. '\n')
        end        
      end        
    
      classifierGrads:div(numPhonemes)
      -- keep loss over entire training data
      totalLoss = totalLoss + loss
      numTotalPhonemes = numTotalPhonemes + numPhonemes
      -- loss for current batch
      loss = loss/numPhonemes
      
      return loss, classifierGrads      
    end
    
    optimMethod(evalLossGrad, classifierParams, optimState)
  
  end
  
  time = (sys.clock() - time) / dataSize
  print('==> time to learn 1 sample = ' .. (time*1000) .. 'ms') 
  totalLoss = totalLoss/numTotalPhonemes
  print('==> loss: ' .. totalLoss)  
  print(confusion)
  print('==> total number of train phonemes (frames): ' .. numTotalPhonemes)
  local accuracy = confusion.totalValid * 100
     
  -- for next epoch
  confusion:zero()
  
  if predFile then predFile:close() end
  if goldFile then goldFile:close() end
  if transFile then transFile:close() end
  
  return totalLoss, accuracy
      
end
  
  
function evalAggregateFrames(allSpects, allTranscripts, allTimes, dataSize, epoch, opt, testOrVal, predFilename)
  local testOrVal = testOrVal or 'test'  
  local predFile, goldFile, transFile
  if predFilename then
    predFile = torch.DiskFile(predFilename .. '.epoch' .. epoch, 'w')
    goldFile = torch.DiskFile(predFilename .. '.gold', 'w')
    transFile = torch.DiskFile(predFilename .. '.trans', 'w')
  end  
  local time = sys.clock()
  classifier:evaluate()
  local shuffle = torch.range(1, dataSize) -- no need to shuffle on test TODO: clean this
  
  local input = torch.Tensor()
  if opt.nGPU > 0 then
    input = input:cuda()
  end
  
  print('\n==> evaluating on ' .. testOrVal .. ' data')
  print('\n==> epoch # ' .. epoch .. ' [batch size = ' .. opt.batchSize .. ']')  
  
  local totalLoss, numTotalPhonemes = 0, 0
  for i = 1,dataSize,opt.batchSize do
    collectgarbage()
    xlua.progress(i, dataSize)
    
    -- get next batch
    local indices = shuffle[{ {i, math.min(i+opt.batchSize-1, dataSize) } }]
    local inputsCPU, _, transcripts, times = nextBatch(indices, allSpects, allTranscripts, allTimes)
    input:resize(inputsCPU:size()):copy(inputsCPU) -- batch size X 1 X freq X input seq length
          
    local loss, numPhonemes = 0, 0    
    
    -- to compute the convolved seq length = ( ((input seq length - 11) / 2 + 1) - 11 ) / 2 + 1, or use the function calSize(input seq length)
    
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
      local predLabels, goldLabels = {}, {}
      
      local prevTarget, curNumFrames, aggregateRepr = 0, 0, torch.zeros(repr[1][k]:nElement())
      if opt.nGPU > 0 then aggregateRepr = aggregateRepr:cuda() end      
      -- iterate over time
      for t = 1, repr:size(timeDim) do 
        -- get label id corresponding to current frame; if frame is outside of transcript (because of padding), will return 0
        local label = getFrameLabel(transcripts[k], times[k], t, opt.reprLayer, opt.convStep)
        if opt.phoneClasses then
          label = phone2class[label]
        end        
        if label then
          local target = label2idx[label]
          
          -- aggregate frames
          if opt.aggregateFrames then
            if prevTarget > 0 then
              if prevTarget == target then
                -- aggregate frames
                curNumFrames = curNumFrames + 1
                aggregateRepr:add(repr[t][k])
              else
                -- found new phoneme; make a forward/backward step on previous one
                -- input to classifier is average over frames
                local classifierInput = aggregateRepr:div(curNumFrames)
                local classifierOut = classifier:forward(classifierInput)
                loss = loss + criterion:forward(classifierOut, target)
                numPhonemes = numPhonemes + 1
                -- get predicted labels to write to file
                if predFile then
                  local _, predIdx =  classifierOut:max(1)
                  predIdx = predIdx:long()[1]
                  local predLabel = idx2label[predIdx]
                  table.insert(predLabels, predLabel)
                  table.insert(goldLabels, label)
                end               
                
                -- update confusion matrix
                confusion:add(classifierOut, target)
                
                -- init for next phoneme
                curNumFrames = 1
                aggregateRepr:copy(repr[t][k])
              end
            else
              -- first frame in the utterance
              curNumFrames = 1
              aggregateRepr:copy(repr[t][k])
            end
            -- update previous target
            prevTarget = target
            
            -- clean last frame
            if t == repr:size(timeDim) and curNumFrames > 0 then
              local classifierInput = aggregateRepr:div(curNumFrames)
              local classifierOut = classifier:forward(classifierInput)
              loss = loss + criterion:forward(classifierOut, target)
              numPhonemes = numPhonemes + 1
              -- get predicted labels to write to file
              if predFile then
                local _, predIdx =  classifierOut:max(1)
                predIdx = predIdx:long()[1]
                local predLabel = idx2label[predIdx]
                table.insert(predLabels, predLabel)
                table.insert(goldLabels, label)
              end               
              
              -- update confusion matrix
              confusion:add(classifierOut, target)
            end
            
          else
          
            local classifierInput = repr[t][k]
            local classifierOut = classifier:forward(classifierInput)
            loss = loss + criterion:forward(classifierOut, target)
            numPhonemes = numPhonemes + 1
            
            -- get predicted labels to write to file
            if predFile then
              local _, predIdx =  classifierOut:max(1)
              predIdx = predIdx:long()[1]
              local predLabel = idx2label[predIdx]
              table.insert(predLabels, predLabel)
              table.insert(goldLabels, label)
            end          
            
            -- update confusion matrix
            confusion:add(classifierOut, target)
          end
          
        elseif opt.aggregateFrames and curNumFrames > 0 then
          -- if found invalid label (e.g. silence), and aggregating frames, and have some frames aggregated, make a forward/backword step
          local target = prevTarget
          local classifierInput = aggregateRepr:div(curNumFrames)
          local classifierOut = classifier:forward(classifierInput)
          loss = loss + criterion:forward(classifierOut, target)
          numPhonemes = numPhonemes + 1
          local outputGrad = criterion:backward(classifierOut, target)
          classifier:backward(classifierInput, outputGrad)
          -- get predicted labels to write to file
          if predFile then
            local _, predIdx =  classifierOut:max(1)
            predIdx = predIdx:long()[1]
            local predLabel = idx2label[predIdx]
            table.insert(predLabels, predLabel)
            table.insert(goldLabels, label)
          end               
          
          -- update confusion matrix
          confusion:add(classifierOut, target) 
          
          -- init for next phoneme
          prevTarget, curNumFrames, aggregateRepr = 0, 0, torch.zeros(repr[1][k]:nElement())
          if opt.nGPU > 0 then aggregateRepr = aggregateRepr:cuda() end                  
        end            
      end
      if predFile then
        predFile:writeString(stringx.join(' ', predLabels) .. '\n')
        goldFile:writeString(stringx.join(' ', goldLabels) .. '\n')
        transFile:writeString(transcripts[k] .. '\n')
      end
      
    end        
  
    classifierGrads:div(numPhonemes)
    -- keep loss over entire training data
    totalLoss = totalLoss + loss
    numTotalPhonemes = numTotalPhonemes + numPhonemes
    -- loss for current batch
    loss = loss/numPhonemes
  
  end
  
  time = (sys.clock() - time) / dataSize
  print('==> time to evaluate 1 sample = ' .. (time*1000) .. 'ms') 
  totalLoss = totalLoss/numTotalPhonemes
  print('==> loss: ' .. totalLoss)  
  print(confusion)
  print('==> total number of ' .. testOrVal .. ' phonemes (frames): ' .. numTotalPhonemes)
  local accuracy = confusion.totalValid * 100
     
  -- for next epoch
  confusion:zero()
  
  if predFile then predFile:close() end
  if goldFile then goldFile:close() end
  if transFile then transFile:close() end
  
  return totalLoss, accuracy  
      
end  
  
  
main(opt)  
