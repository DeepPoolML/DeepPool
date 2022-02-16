import torch
from torch import nn, Tensor
import numpy as np
from einops import rearrange
import random
from typing import List, Tuple
from einops import repeat
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from clusterClient import ClusterClient
from jobDescription import TrainingJob

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)

def init_random_seed(seed, gpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if gpu:
        torch.backends.cudnn.deterministic = True

# from https://huggingface.co/transformers/_modules/transformers/modeling_utils.html
def get_module_device(parameter: nn.Module):
    try:
        return next(parameter.parameters()).device
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device

def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) * scale_factor

    if mask is not None:
        assert mask.shape == scaled_dot_prod.shape[2:]
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)

class TransformerBlockA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dropout=0.1):
        super().__init__()                
        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.drop = nn.Dropout(dropout)
 
        self.norm_1 = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        y = self.norm_1(self.drop(self.mhsa(x, mask)) + x)
        return y


class TransformerBlockB(nn.Module):
    def __init__(self, dim, dim_linear_block=1024, dropout=0.1):
        super().__init__()
        self.norm_2 = nn.LayerNorm(dim)
        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            nn.ReLU(), #activation(),  # nn.ReLU or nn.GELU
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, y):
        out = self.norm_2(self.linear(y) + y)
        return out


class TransformerBlock(nn.Module):
    """
    Vanilla transformer block from the original paper "Attention is all you need"
    Detailed analysis: https://theaisummer.com/transformer/
    """

    def __init__(self, dim, heads=8, dim_head=None,
                 dim_linear_block=1024, dropout=0.1):
        """
        Args:
            dim: token's vector length
            heads: number of heads
            dim_head: if none dim/heads is used
            dim_linear_block: the inner projection dim
            dropout: probability of droppping values
            mhsa: if provided you can change the vanilla self-attention block
            prenorm: if the layer norm will be applied before the mhsa or after
        """
        super().__init__()
        self.blockA = TransformerBlockA(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        paramsA = {"dim": dim, "heads": heads, "dim_head": dim_head}
        cs.GeneralLayer(self.blockA, "TransBlockA", paramsA, mustTrace=True)

        self.blockB = TransformerBlockB(dim, dim_linear_block=dim_linear_block, dropout=dropout)
        paramsB = {"dim": dim, "dim_linear_block": dim_linear_block}
        cs.GeneralLayer(self.blockA, "TransBlockB", paramsB, mustTrace=True)

    def forward(self, x, mask=None):
        y = self.blockA(x)
        out = self.blockB(y)
        return out

# def GeneralLayer(self, module, name, params, custom_previous_layers: list = None):

class TransformerEncoder(nn.Module):
    def __init__(self, dim, blocks=6, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.0):
        super().__init__()
        self.block_list = [TransformerBlock(dim, heads, dim_head,
                                            dim_linear_block, dropout) for _ in range(blocks)]
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class prepInput(nn.Module):
    def __init__(self,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 dim=512,
                 dropout=0.0):
        super().__init__()
        self.p = patch_dim
        # tokens = number of patches
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim ** 2)
        self.dim = dim

        # Projection and pos embeddings
        self.project_patches = nn.Linear(self.token_dim, self.dim)
        self.emb_dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, self.dim))

    def forward(self, img, mask=None):
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.p, patch_y=self.p)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.emb_dropout(img_patches)
        return patch_embeddings

class prepOutput(nn.Module):
    def __init__(self, classification):
        super().__init__()
        self.classification = classification
    def forward(self, y):
        if self.classification:
            return y[:, 0, :]
        else:
            return y[:, 1:, :]

class ViT(nn.Module):
    def __init__(self, *,
                 img_dim,
                 in_channels=3,
                 patch_dim=16,
                 num_classes=10,
                 dim=512,
                 blocks=6,
                 heads=4,
                 dim_linear_block=1024,
                 dim_head=None,
                 dropout=0.0, transformer=None, classification=True):
        """
        Minimal re-implementation of ViT
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token that we will index in the final classification layer
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f'patch size {patch_dim} not divisible by img dim {img_dim}'
        # self.p = patch_dim
        self.classification = classification
        # # tokens = number of patches
        self.dim = dim
        self.dim_head = (int(self.dim / heads)) if dim_head is None else dim_head

        self.prep = prepInput(img_dim, in_channels, patch_dim, dim, dropout)
        prepParams = {"img_dim": img_dim, "in_channels": in_channels, "patch_dim": patch_dim, "dim": dim, "dropout": dropout}
        cs.GeneralLayer(self.prep, "prep", prepParams, mustTrace=True)

        self.transformer = TransformerEncoder(self.dim, blocks=blocks, heads=heads,
                                                dim_head=self.dim_head,
                                                dim_linear_block=dim_linear_block,
                                                dropout=dropout)

        self.prepOutput = prepOutput(self.classification)
        cs.GeneralLayer(self.prepOutput, "prepOut", {}, mustTrace=True)

        if self.classification:
            self.mlp_head = cs.Linear(self.dim, num_classes)


    def forward(self, img, mask=None):
        # # Create patches
        # # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        # img_patches = rearrange(img,
        #                         'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
        #                         patch_x=self.p, patch_y=self.p)

        # batch_size, tokens, _ = img_patches.shape

        # # project patches with linear layer + add pos emb
        # img_patches = self.project_patches(img_patches)

        # img_patches = torch.cat((expand_to_batch(self.cls_token, desired_size=batch_size), img_patches), dim=1)

        # # add pos. embeddings. + dropout
        # # indexing with the current batch's token length to support variable sequences
        # img_patches = img_patches + self.pos_emb1D[:tokens + 1, :]
        # patch_embeddings = self.emb_dropout(img_patches)
        patch_embeddings = self.prep(img)

        

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)

        # we index only the cls token for classification. nlp tricks :P
        return self.mlp_head(self.prepOutput(y))



def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False):
    global cs
    cs = CostSim(None, netBw=netBw, verbose=True, gpuProfileLoc="vitLayerGpuProfileA100.txt")
    # model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=1000, dim=512, blocks=12, dim_linear_block=3072)
    model = ViT(img_dim=256, in_channels=3, patch_dim=32, num_classes=1000, dim=1024, blocks=24, heads=16, dim_linear_block=4092)
    cs.printAllLayers(silent=True)
    cs.computeInputDimensions((3,256,256))
    # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)

    # if dataParallelBaseline:
    #     dpIterUsec, dpFpUsec, dpBpUsec = profiler.benchModel(model, (3, 299, 299), int(globalBatch / gpuCount))
    #     print("(DP baseline) whole model bench: %.1f ms (fp: %.1f, bp: %.1f)" % (dpIterUsec / 1000, dpFpUsec / 1000, dpBpUsec / 1000))

    job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
    cs.to_dot(simResultFilename, globalBatch)
    # cs.to_gpuTimeline("Inception v3, Burst Parallel", maxGpusUsed, dataParallelBaseline)
    jobInJson = job.dumpInJSON()

    # for rank in range(gpuCount):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    # job2 = TrainingJob("test", None, None, 0, 0, "")
    # job2.loadJSON(jobInJson)
    # assert(jobInJson == job2.dumpInJSON())
    # print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)
    
    if maxGpusUsed > 8:
        print("maxGpusUsed: ", maxGpusUsed, " is bigger than 8. Can't schedule this job.")
        exit(-1)
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = "ViT_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        cc.submitTrainingJob(jobName, jobInJson)

def runStrongScalingBench():
    global cs
    netBw = 2.66E5
    cs = CostSim(None, netBw=netBw, verbose=False)
    inputSize = (3,256,256)
    model = ViT(img_dim=256, in_channels=3, patch_dim=16, num_classes=1000, dim=512)

    fakeInputSize = (16,3,256,256)
    fakeInput = torch.zeros(fakeInputSize)
    traced = torch.jit.trace(model, fakeInput)
    torch.jit.save(traced, "modules/vit.pt")
    
    print("Model: ", "Vit(256, 3, 16, 1000, 512")
    print("BatchSize  iterMs    fpMs    bpMs")
    for batchSize in [2 ** exp for exp in range(1, 9)]:
        assert False
        # iterTime, fpTime, bpTime = profiler.benchModel(model, inputSize, batchSize)
        # print(" %8d  %6.1f  %6.1f  %6.1f" %
            # (batchSize, iterTime / 1000, fpTime / 10000, bpTime / 1000))


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        gpuCount = int(sys.argv[1])
        globalBatchSize = int(sys.argv[2])
        simResultFilename = "%s_%s_b%d_sim.data" % ("inception", "DP", globalBatchSize)
        main(gpuCount, globalBatchSize, dataParallelBaseline=True)
    elif len(sys.argv) == 4:
        gpuCount = int(sys.argv[1])
        globalBatchSize = int(sys.argv[2])
        amplificationLimit = float(sys.argv[3])
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % ("inception", "MP", globalBatchSize, amplificationLimit)
        main(gpuCount, globalBatchSize, amplificationLimit, simResultFilename = simResultFilename)#, netBw = 1.25E4)
    elif len(sys.argv) == 2:
        print("Run all configs")
        # runAllConfigs("inceptionV3", sys.argv[1])
    elif len(sys.argv) == 1:
        runStrongScalingBench()
    else:
        print("Wrong number of arguments.\nUsage: ")
