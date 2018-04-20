stringx = require 'pl.stringx'
tds = require 'tds'



function getPhoneClasses(filename)
  local phone2class, classes = {}, {}
  for line in io.lines(filename) do
    local phone, class = unpack(line:split(' '))
    phone2class[phone] = class
    if not classes[class] then
      table.insert(classes, class)
    end
  end
  return phone2class, classes
end


function getLabels(dirPath)
  local dbTrans = lmdb.env { Path = dirPath .. '/trans', Name = 'trans' }
  dbTrans:open()
  local readerTrans = dbTrans:txn(true)
  local dataSize = dbTrans:stat()['entries']
  local label2idx, idx2label = {}, {}
  for i = 1,dataSize do
    local labels = readerTrans:get(i):split(' ')
    for t, label in pairs(labels) do
      if not label2idx[label] then
        idx2label[#idx2label+1] = label
        label2idx[label] = #idx2label        
      end
    end
  end
  dbTrans:close()
  return label2idx, idx2label  
end


function getFreq(dirPath)
  local dbSpect = lmdb.env { Path = dirPath .. '/spect', Name = 'spect' }
  dbSpect:open()
  local readerSpect = dbSpect:txn(true)
  local tensor = readerSpect:get(1):float()
  local freq = tensor:size(1)
  dbSpect:close()
  return freq
end


function getSplitDBs(dirPath)
  local dbSpect = lmdb.env { Path = dirPath .. '/spect', Name = 'spect' }
  local dbTrans = lmdb.env { Path = dirPath .. '/trans', Name = 'trans' }
  local dbTimes = lmdb.env { Path = dirPath .. '/times', Name = 'times' }

  dbSpect:open()
  local dataSize = dbSpect:stat()['entries']
  local readerSpect = dbSpect:txn(true)
  local tensor = readerSpect:get(1):float()
  local freq = tensor:size(1)
  
  dbSpect:close()
  
  return dbSpect, dbTrans, dbTimes, dataSize
end

-- TODO merge with getSplitDBs
function getSplitDBsUttids(dirPath)
  local dbSpect = lmdb.env { Path = dirPath .. '/spect', Name = 'spect' }
  local dbTrans = lmdb.env { Path = dirPath .. '/trans', Name = 'trans' }
  local dbUttids = lmdb.env { Path = dirPath .. '/uttid', Name = 'uttid' }

  dbSpect:open()
  local dataSize = dbSpect:stat()['entries']
  local readerSpect = dbSpect:txn(true)
  local tensor = readerSpect:get(1):float()
  local freq = tensor:size(1)
  
  dbSpect:close()
  
  return dbSpect, dbTrans, dbUttids, dataSize
end


function loadData(dbSpect, dbTrans, dbTimes)
  local tensors = tds.Vec()
  --local targets = {}
  local transcripts = {}
  local times = {}

  local freq = 0  
  
  dbSpect:open(); local readerSpect = dbSpect:txn(true) 
  dbTrans:open(); local readerTrans = dbTrans:txn(true)
  dbTimes:open(); local readerTimes = dbTimes:txn(true)
  
  local size = dbSpect:stat()['entries']

  -- read out all the data and store in lists
  for x = 1, size do
    local tensor = readerSpect:get(x):float()
    local transcript = readerTrans:get(x)
    local curTimes = readerTimes:get(x):long()

    freq = tensor:size(1)

    tensors:insert(tensor)
    --table.insert(targets, self.mapper:encodeString(transcript))
    table.insert(transcripts, transcript)
    table.insert(times, curTimes)
  end

  readerSpect:abort(); dbSpect:close()
  readerTrans:abort(); dbTrans:close()
  readerTimes:abort(); dbTimes:close()
  
  return tensors, transcripts, times
end

-- TODO merge with loadData
function loadDataUttids(dbSpect, dbTrans, dbUttids)
  local tensors = tds.Vec()
  --local targets = {}
  local transcripts = {}
  local uttids = {}

  local freq = 0  
  
  dbSpect:open(); local readerSpect = dbSpect:txn(true) 
  dbTrans:open(); local readerTrans = dbTrans:txn(true)
  dbUttids:open(); local readerUttids = dbUttids:txn(true)
  
  local size = dbSpect:stat()['entries']

  -- read out all the data and store in lists
  for x = 1, size do
    local tensor = readerSpect:get(x):float()
    local transcript = readerTrans:get(x)
    local curUttid = readerUttids:get(x)

    freq = tensor:size(1)

    tensors:insert(tensor)
    --table.insert(targets, self.mapper:encodeString(transcript))
    table.insert(transcripts, transcript)
    table.insert(uttids, curUttid)
  end

  readerSpect:abort(); dbSpect:close()
  readerTrans:abort(); dbTrans:close()
  readerUttids:abort(); dbUttids:close()
  
  return tensors, transcripts, uttids
end


function nextBatch(indices, spects, transcripts, times)
  local batchTensors = tds.Vec()
  --local targets = {}
  local batchTranscripts = {}
  local batchTimes = {}

  local maxLength = 0
  local freq = 0  
  
  local size = indices:size(1) 
  local batchSizes = torch.Tensor(#indices)

  -- reads out a batch and store in lists
  for x = 1, size do
    local ind = indices[x]
    local tensor = spects[ind]
    local transcript = transcripts[ind]
    local curTimes = times[ind]

    freq = tensor:size(1)
    batchSizes[x] = tensor:size(2)
    if maxLength < tensor:size(2) then maxLength = tensor:size(2) end -- find the max len in this batch

    batchTensors:insert(tensor)
    --table.insert(targets, self.mapper:encodeString(transcript))
    table.insert(batchTranscripts, transcript)
    table.insert(batchTimes, curTimes)
  end

  local batchInputs = torch.Tensor(size, 1, freq, maxLength):zero()
  for ind, tensor in ipairs(batchTensors) do
    batchInputs[ind][1]:narrow(2, 1, tensor:size(2)):copy(tensor)
  end
  
  --return inputs, targets, sizes, transcripts, times
  return batchInputs, sizes, batchTranscripts, batchTimes
end

function nextBatchUttids(indices, spects, transcripts, uttids)
  local batchTensors = tds.Vec()
  --local targets = {}
  local batchTranscripts = {}
  local batchUttids = {}

  local maxLength = 0
  local freq = 0  
  
  local size = indices:size(1) 
  local batchSizes = torch.Tensor(#indices)

  -- reads out a batch and store in lists
  for x = 1, size do
    local ind = indices[x]
    local tensor = spects[ind]
    local transcript = transcripts[ind]
    local curUttid = uttids[ind]

    freq = tensor:size(1)
    batchSizes[x] = tensor:size(2)
    if maxLength < tensor:size(2) then maxLength = tensor:size(2) end -- find the max len in this batch

    batchTensors:insert(tensor)
    --table.insert(targets, self.mapper:encodeString(transcript))
    table.insert(batchTranscripts, transcript)
    table.insert(batchUttids, curUttid)
  end

  local batchInputs = torch.Tensor(size, 1, freq, maxLength):zero()
  for ind, tensor in ipairs(batchTensors) do
    batchInputs[ind][1]:narrow(2, 1, tensor:size(2)):copy(tensor)
  end
  
  --return inputs, targets, sizes, transcripts, uttids
  return batchInputs, sizes, batchTranscripts, batchUttids
end


--[[
find frame label (a phoneme string)

transcript: a string of phonemes
times: a tensor of integers representing phoneme end times in the input data
t: an index of current frame that is input to the classifier (and output of deep speech model)
layer: a name of the layer whose output is input to the classifier
convStepSize: step size of convolution in time dimension
sampleRate: sampling rate of the original audio
spectStride: stride of the spectrogram
spectwindowSize: window size of the spectrogram
ignoreSilence: whether to ignore begin/end silence 
--]] 
function getFrameLabel(transcript, times, t, layer, convStepSize, sampleRate, spectStride, spectWindowSize, ignoreSilence)
  local sampleRate = 16000 or sampleRate
  local spectStride = 0.01 or spectStride
  local spectWindowSize = 0.02 or spectWindowSize
  local ignoreSilence = ignoreSilence or true
  local convStepSize = convStepSize or 2
  
  local deepSpeechInputFrame -- index of input frame to the deep speech model
  if layer == 'cnn' or layer == 'cnn2' or stringx.startswith(layer, 'rnn') then
    deepSpeechInputFrame = ( ((t-1)*convStepSize+11) - 1 )*convStepSize+11 
  elseif layer == 'cnn1' then 
    deepSpeechInputFrame = (t-1)*convStepSize+11
  elseif layer == 'input' then
    deepSpeechInputFrame = t
  else  
    error('Unsupported layer ' .. layer .. ' in getFrameLabelIdx')
  end   
    
  local windowStart = deepSpeechInputFrame*sampleRate*spectStride
  local windowMiddle = math.floor(windowStart + spectWindowSize*sampleRate/2)
  -- find end time
  for i = 1,times:size(1) do
    if windowMiddle < times[i] then
      -- if ignore begin/end silence
      if ignoreSilence and (i == 1 or i == times:size(1)) then
        return nil
      else
        return transcript:split(' ')[i]
      end
    end
  end
  -- if frame is out of times, it will be ignored
  return nil
end  


function getWindowedInput(windowSize, repr, t, k, timeDim)
  local windowedInput = torch.zeros(2*windowSize+1, repr[t][k]:nElement())
  -- TODO fix this to use repr to create new zero tensor
  windowedInput = windowedInput:cuda()
  -- TODO vectorize
  for w = 1, 2*windowSize+1 do
    local curFrameId = t-windowSize-1+w
    if curFrameId >= 1 and curFrameId <= repr:size(timeDim) then      
      windowedInput[w] = repr[curFrameId][k]
    end
  end
  return windowedInput:view(windowedInput:nElement())  
end


function writeMatrixToFile(mat, file, sep)  
  assert(mat:dim() == 2, 'wrong matrix dimension: ' .. mat:dim())  
  local sep = sep or ' ' 
  local f = assert(io.open(file, 'w'))
  for i = 1, mat:size(1) do
    for j = 1, mat:size(2) do
      f:write(mat[i][j])
      if j == mat:size(2) then
        f:write('\n')
      else
        f:write(sep)
      end
    end
  end  
  f:close()
end