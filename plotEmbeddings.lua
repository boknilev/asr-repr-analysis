manifold = require 'manifold'
require 'gnuplot'
require 'image'
--require 'cutorch'
stringx = require 'pl.stringx'

-- uses code from: https://github.com/clementfarabet/manifold/blob/master/demos/demo_tsne.lua


local cmd = torch.CmdLine()
cmd:option('-figFile', 'tsne.png', 'File to save gold labels corresponding to each embedding')
cmd:option('-title', 'tsne', 'Plot title')
cmd:option('-centroidsFile', '/data/sls/scratch/belinkov/asr/prediction/repr-input/timit_train.input.centroids.t7', 'File to save centroids')
cmd:option('-clusteringFile', '/data/sls/scratch/belinkov/asr/prediction/repr-input/timit_train.input.clustering.t7', 'File to save clustering as t7 file')
cmd:option('-perplexity', 30, 't-sne perplexity')
cmd:option('-pca', 50, 't-sne pca')
cmd:option('-use_bh', false, 't-sne use Barnes Hut')
cmd:option('-theta', 0.5, 't-sne theta')
cmd:option('-max_iter', 1000, 't-sne max iteration')
cmd:option('-ndims', 2, 'for plotting')
cmd:option('-majorityThreshold', 0.0, 'Minimum fraction of majority label examples')
cmd:option('-phonesFile', '', 'File with list of phones to plot')

local opt = cmd:parse(arg)
print(opt)


-- TODO
--[[
assign cluster label by majority vote - done
tsne centroids with their majority label - done
color centroids by cluster purity or something like that
--]]


function assignMajorityLabel(clustering, centroids)  
  -- count
  local cluster2label2count = {}
  for i = 1,#clustering do 
    local label, cluster = clustering[i]['label'], clustering[i]['cluster']
    if cluster2label2count[cluster] then
      if cluster2label2count[cluster][label] then
        cluster2label2count[cluster][label] = cluster2label2count[cluster][label] + 1
      else
        cluster2label2count[cluster][label] = 1
      end
    else
      cluster2label2count[cluster] = {label = 1}      
    end
  end
  
  local numClusters = centroids['centroids']:size(1)  
  -- find majority
  centroids['majorityLabelCounts'] = torch.zeros(numClusters)
  centroids['clusterSizes'] = torch.zeros(numClusters)
  centroids['majorityLabels'] = {}
  for i = 1,numClusters do 
    centroids['clusterSizes'][i] = 0
    if cluster2label2count[i] then
      local maxCount, maxLabel = 0
      for l, c in pairs(cluster2label2count[i]) do
        centroids['clusterSizes'][i] = centroids['clusterSizes'][i] + c
        if c > maxCount then
          maxCount, maxLabel = c, l
        end
      end
      centroids['majorityLabels'][i] = maxLabel
      centroids['majorityLabelCounts'][i] = maxCount
    else
      centroids['majorityLabels'][i] = 'nil' -- no examples assigned to this cluster
      centroids['majorityLabelCounts'][i] = 0
    end
  end
    
  return centroids
end


-- compute
--[[
function computeClusterVariances(clustering, centroids)  
  
end
--]]


-- function to show a group scatter plot:
local function show_scatter_plot(fileName, title, mapped_x, labels, idx2label, label2idx)

   -- count label sizes:
   local K = #idx2label
   print('K: ' .. K)
   local cnts = torch.zeros(K)
   print(idx2label)
   print(label2idx)
   for n = 1,#labels do
     -- TODO why +1?
      --cnts[labels[n] + 1] = cnts[labels[n] + 1] + 1
      print(labels[n])
      print(label2idx[labels[n]])
      if label2idx[labels[n]] then
        cnts[label2idx[labels[n]]] = cnts[label2idx[labels[n]]] + 1
      end
   end

   -- separate mapped data per label:
   mapped_data = {}
   for k = 1,K do
      mapped_data[k] = {idx2label[k], torch.Tensor(cnts[k], opt.ndims), '+'}
   end
   local offset = torch.Tensor(K):fill(1)
   for n = 1,#labels do
      if label2idx[labels[n]] then
        mapped_data[label2idx[labels[n]]][2][offset[label2idx[labels[n]]]]:copy(mapped_x[n])
        offset[label2idx[labels[n]]] = offset[label2idx[labels[n]]] + 1
      end
   end
   -- remove missing points with no counts
   new_mapped_data = {}
   for k = 1,K do
      if mapped_data[k][2]:dim() ~= 0 then
        table.insert(new_mapped_data, mapped_data[k])
      else
        table.insert(new_mapped_data, {mapped_data[k][1], torch.zeros(1, 2), 'with points pointsize 0'})
      end      
   end
   mapped_data = new_mapped_data

   -- show results in scatter plot:  
   gnuplot.figure()
   gnuplot.pngfigure(fileName); gnuplot.grid(false); gnuplot.title(title)
   print(mapped_data)
   gnuplot.plot(mapped_data)
   --gnuplot.movelegend('left', 'top')
   --gnuplot.raw('set key bmargin center horizontal')
   gnuplot.raw('unset key')
   gnuplot.raw('unset xtics; unset ytics')
   gnuplot.plotflush()
   
   gnuplot.figure()
   gnuplot.pngfigure('legend.png')
   local entries = {}
   for k = 1,K do
     entries[k] = {idx2label[k], torch.zeros(1,2), '+'}
   end
   gnuplot.plot(entries)
   --gnuplot.raw('set key bmargin center horizontal')
   gnuplot.raw('set key bmargin center vertical maxrows 4 box 3 width -8')
   gnuplot.raw('unset xtics; unset ytics')
   --gnuplot.raw('set size ratio 0.5 2,2')
   --gnuplot.raw('set term png size 1600,800') --800 pixels by 400 pixels
   gnuplot.plotflush()
end


print('Loading clustering')
clustering = torch.load(opt.clusteringFile)
print('Loading centroids')
centroids = torch.load(opt.centroidsFile)
timer = torch.Timer()
print('Assigning majority label')
centroids = assignMajorityLabel(clustering, centroids)


print('Filtering clusters with majority label ratio < ' .. opt.majorityThreshold)
thresholdMask = torch.ge(torch.cdiv(centroids['majorityLabelCounts'], centroids['clusterSizes']), opt.majorityThreshold)
print(thresholdMask)
filteredCentroids = centroids['centroids']:index(1, torch.range(1, thresholdMask:nElement())[thresholdMask]:long())
filteredMajorityLabels = {}
for i = 1,#centroids['majorityLabels'] do
  if thresholdMask[i] == 1 then
    table.insert(filteredMajorityLabels, centroids['majorityLabels'][i])
  end
end


labels = {}
if path.exists(opt.phonesFile) then
  for line in io.lines(opt.phonesFile) do
    table.insert(labels, stringx.strip(line))
  end
else
  for i, l in pairs(filteredMajorityLabels) do
    table.insert(labels, l)
  end
end
table.sort(labels)
idx2label, label2idx = {}, {}
for i, l in pairs(labels) do
  if not label2idx[l] then
    idx2label[#idx2label+1] = l
    label2idx[l] = #idx2label    
  end
end
print('labels:'); print(labels);
print('label2idx:'); print(label2idx);
print('idx2label:'); print(idx2label);


--print(filteredCentroids)
--print(filteredMajorityLabels)
print('After filtering, left with ' .. #filteredMajorityLabels .. ' clusters')


print('Performing t-SNE')
timer = torch.Timer()
tsneOpts = {ndims = 2, perplexity = opt.perplexity, pca = opt.pca, use_bh = opt.use_bh, max_iter = opt.max_iter}
--mappedCentroids = manifold.embedding.tsne(centroids['centroids']:double(), tsneOpts)
mappedCentroids = manifold.embedding.tsne(filteredCentroids:double(), tsneOpts)
print('Successfully performed t-SNE in ' .. timer:time().real .. ' seconds.')

print('Plotting scatter plot')
--gnuplot.plot(mappedCentroids, '+')
--show_scatter_plot(opt.figFile, opt.title, mappedCentroids, centroids['majorityLabels'], idx2label, label2idx)
show_scatter_plot(opt.figFile, opt.title, mappedCentroids, filteredMajorityLabels, idx2label, label2idx)



