-- modified from David Harwath's clusterCCGroundingsPaired.lua script 


require 'json'
--require 'unsup'
unsup = {}
require 'kmeans'
--require 'nn'
--require 'cunn'
--require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')


local cmd = torch.CmdLine()
cmd:option('-embFile', 'emb.t7', 'File with embeddings extracted from Deep Speech model')
cmd:option('-labelFile', 'lbl.t7', 'File with gold labels corresponding to each embedding')
cmd:option('-centroidOut', '/data/sls/scratch/belinkov/asr/prediction/repr-input/timit_train.input.centroids.t7', 'File to save centroids')
cmd:option('-clusteringOut', '/data/sls/scratch/belinkov/asr/prediction/repr-input/timit_train.input.clustering.t7', 'File to save clustering as t7 file')
cmd:option('-clusteringJsonOut', '/data/sls/scratch/belinkov/asr/prediction/repr-input/timit_train.input.clustering.json', 'File to save clustering as json file')
cmd:option('-nGPU', 1, 'Number of GPUs, set -1 to use CPU')
cmd:option('-batchSize', 10000, 'Batch size')
cmd:option('-numIters', 40, 'Number of K-means iterations')
cmd:option('-numClusters', 500, 'Number of K-means clusters')
cmd:option('-scoreThresh ', 0.5, '???')

local opt = cmd:parse(arg)
print(opt)

if opt.nGPU > 0 then require 'cutorch' end

-- globals
batchSize = opt.batchSize


function verifyClustering(embeddings, centroids, labels, distances, N)
  local epsilon = 0.001
  for i = 1,N do
    local embedding = embeddings[i]
    local lowestDist = math.huge
    local lowestDistIdx = 0
    for j = 1,(#centroids)[1] do
      local centroid = centroids[j]
      local distance = centroid:dist(embedding)
      if distance < lowestDist then
        lowestDist = distance
        lowestDistIdx = j
      end
    end
    --print(('(Assignment) Label: %d \t Distance: %.3f'):format(labels[i], distances[i]))
    --print(('(Verification) Label: %d \t Distance: %.3f'):format(lowestDistIdx, lowestDist))
    assert(lowestDistIdx == labels[i])
    assert(math.abs(distances[i] - lowestDist) < (epsilon * lowestDist))
  end
end

function assignPointsToCentroids(x, centroids)
  local batchsize = math.min(batchSize, (#x)[1])
  local k = (#centroids)[1]
  -- resize data
  local k_size = x:size()
  k_size[1] = k
  if x:dim() > 2 then
    x = x:reshape(x:size(1), x:nElement()/x:size(1))
  end

  -- some shortcuts
  local sum = torch.sum
  local max = torch.max
  local pow = torch.pow

  -- dims
  local nsamples = (#x)[1]
  local ndims = (#x)[2]

  -- sums of squares
  local c2 = sum(pow(centroids,2),2)*0.5

  -- init some variables
  local summation = x.new(k,ndims):zero()
  local counts = x.new(k):zero()
  local loss = 0
  local labels = x.new(nsamples):zero()
  local distances = x.new(nsamples):zero()
  -- process batch
  for i = 1,nsamples,batchsize do
     -- indices
     local lasti = math.min(i+batchsize-1,nsamples)
     local m = lasti - i + 1

     -- k-means step, on minibatch
     local batch = x[{ {i,lasti},{} }]
     local x2 = sum(pow(batch,2),2)
     local batch_t = batch:t()
     local tmp = centroids * batch_t
     for n = 1,(#batch)[1] do
        tmp[{ {},n }]:add(-1,c2)
     end
     local val,batch_labels = max(tmp,1)
     local labels_sub = labels:sub(i, lasti)
     labels_sub:copy(batch_labels[1])
     local distances_sub = distances:sub(i, lasti)
     local batch_distances = x2 - 2*val:t()
     batch_distances:sqrt()
     distances_sub:copy(batch_distances)
  end
  return labels, distances
end

function getUttidFromPath(audioPath)
  local toks = string.split(audioPath, '/')
  local filename = toks[#toks]
  local filenameToks = string.split(filename, '%.')
  local uttid = filenameToks[1]
  return uttid
end


print('Starting to cluster groundings...')
local timer = torch.Timer()
print('Loading everything...')
local embeddings = torch.load(opt.embFile)
-- reshape to num samples X embedding size
embeddings = embeddings:transpose(1, 2)
--small data
--embeddings = embeddings[{ {1, 750000}, {}}]
--print(embeddings)

local dataLabels = torch.load(opt.labelFile)
--local groundingInfo = torch.load(opt.inputDir .. '/' .. 'ccData.t7')

print(('Loaded embeddings and labels for %d \t(Time %.3f)'):format(#dataLabels, timer:time().real))

print('Normalizing embeddings...')
timer:reset()
local embMean = embeddings:mean(1)
local embStd = embeddings:std(1)
embeddings:add(-1 * embMean:expandAs(embeddings))
embeddings:cdiv(embStd:expandAs(embeddings))

print(string.format('Done normalizing, time taken: %f s', timer:time().real))

collectgarbage()
print('Beginning k-means clustering...')
timer:reset()
if opt.nGPU > 0 then embeddings = embeddings:cuda() end
local centroids, counts = unsup.kmeans(embeddings, opt.numClusters, opt.numIters, batchSize, nil, true)
print(string.format('Done clustering, time taken: %f s', timer:time().real))
print('Assigning groundings to cluster centroids...')
timer:reset()
labels, distances = assignPointsToCentroids(embeddings, centroids)
print('Verifying correctness of assignments...')
verifyClustering(embeddings, centroids, labels, distances, 100)
print('Assignments look good')
print(string.format('Done assigning, time taken: %f s', timer:time().real))

print('Saving centroid info...')
centroidInfo = {}
centroidInfo['centroids'] = centroids:double()
centroidInfo['counts'] = counts:double()
centroidInfo['origEmbMean'] = embMean
centroidInfo['origEmbStd'] = embStd
torch.save(opt.centroidOut, centroidInfo)
centroidInfo = nil
collectgarbage()

print('Saving clustering assignments and scores in info data struct...')
local info = {}
timer:reset()
for index = 1,embeddings:size(1) do
  local datum = {}
  datum['label'] = dataLabels[index]
  datum['cluster'] = labels[index]
  datum['score'] = distances[index]
  table.insert(info, datum)
end
print(('Done organizing\t(Time %.3f), saving data...'):format(timer:time().real))
collectgarbage()
--print(groundingInfo[1])
--json.save(opt.clusteringJsonOut, info)
print('Saving info...')
torch.save(opt.clusteringOut, info)
print('Done.')
