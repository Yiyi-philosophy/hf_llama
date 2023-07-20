
## ===================== ##
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import transformers

# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__

def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    # a = 8 #Alpha value
    a = args.scale
    #The method is just these three lines
    max_position_embeddings = max_position_embeddings * a
    
    base = base * a ** (dim / (dim-2)) #Base change formula

    old_init(self, dim, max_position_embeddings, base, device)
    
def ntk_dym_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    # a = 8 #Alpha value
    a = args.scale
    curr_len = 0
    # dynamic
    a = (a * curr_len / max_position_embeddings) - (a - 1)
    
    #The method is just these three lines
    max_position_embeddings = max_position_embeddings * a
    
    base = base * a ** (dim / (dim-2)) #Base change formula

    old_init(self, dim, max_position_embeddings, base, device)

def pi_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    # scale = 2 # Interpretation value
    scale = args.scale
    max_position_embeddings = max_position_embeddings * scale
    
    old_init(self, dim, max_position_embeddings, base, device)

    
def pi_set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    scale = args.scale # Interpretation value
    print("Interpretation:", scale)
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
    t = t * 1 /scale
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    # Different from paper, but it uses a different permutation in order to obtain the same calculation
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
    self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)



'''
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
'''


    
def main(meth, scale):
    # meth = "standard"
    # meth  = "pi"
    if meth == "ntk":
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_scaled_init
        
    elif meth == "pi":
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = pi_scaled_init
        transformers.models.llama.modeling_llama.LlamaRotaryEmbedding._set_cos_sin_cache = pi_set_cos_sin_cache
        
        
    # model_path = "TheBloke/OpenAssistant-SFT-7-Llama-30B-HF"
    # model_path = "/data/lzhani/prompt_compression/llama/models_hf/7B"
    model_path = "elinas/llama-7b-hf-transformers-4.29"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

    generation_config = GenerationConfig(
        temperature=0.0,
        top_k=20,
        repetition_penalty=1.2,
    )

    #@title
    prompt_full = '''
    You are given this machine learning research paper, please read it carefully and answer the follow up question.

    === BEGIN ===

    2306.15595v2 [cs.CL] 28 Jun 2023

    arXiv

    EXTENDING CONTEXT WINDOW OF LARGE LAN-
    GUAGE MODELS VIA POSITION INTERPOLATION

    Shouyuan Chen Sherman Wong Liangjian Chen  Yuandong Tian
    Meta Platforms Inc.
    {chenshouyuan, shermanwong, cli, yuandong}@meta . com

    1 INTRODUCTION

    Large language models (LLMs) typically come with a pre-defined context window size. For exam-
    ple, inputs to LLaMA models (Touvron et al., 2023) must be fewer than 2048 tokens. This pre-set
    context window limit is frequently exceeded in applications such as conducting long conversations,
    summarizing long documents, or executing long-term planning. For these applications, LLMs with
    longer context windows are preferred. However, training an LLM from scratch with long context
    windows requires significant investments. This naturally leads to a question: Can we extend the
    context window of an existing pre-trained LLM?

    One straightforward approach is to fine-tune an existing pre-trained Transformer with a longer con-
    text window. However, empirically, we found that models trained this way adapt to long context
    windows very slowly. After training for more than 10000 batches, the effective context window
    saw a minimal increase, moving from 2048 to 2560 (Table 4). This suggests that such method is
    inefficient for extending to substantially longer context windows.

    While certain techniques such as ALiBi (Press et al., 2022) and LeX (Sun et al., 2022) enable length
    extrapolation of Transformers, i.e. train on short context windows and inference on longer ones,
    many existing pre-trained LLMs, including LLaMA (Touvron et al., 2023), use positional encodings
    that have weak extrapolation properties (e.g., RoPE (Su et al., 2021)). Therefore, the applicability
    of these techniques for extending the context window sizes of such LLMs remains limited.

    In this work, we introduce Position Interpolation to enable context window extensions for certain
    existing pre-trained LLMs, including LLaMA. The key idea is, instead of extrapolation, we directly
    down-scale the position indices so that the maximum position index matches the previous context
    window limit in the pre-training stage. See Figure 1 for an illustration. In other words, to accom-
    modate more input tokens, we interpolate the position encodings at neighboring integer positions,
    utilizing the fact that position encodings can be applied on non-integer positions, as opposed to
    extrapolating outside the trained positions, which may lead to catastrophic values. We verify our
    approach theoretically, by showing that the interpolated attention score has a much smaller upper

    bound (~ 600x smaller in LLaMA 7B setting) than the extrapolated one, and is thus much more
    stable. Therefore, interpolated position encodings are easier for the model to adapt.

    Empirically, we found that Position Interpolation is highly effective and efficient, requiring only a
    very short period of fine-tuning for the model to fully adapt to greatly extended context windows.
    We present experimental results for extending the context window to up to 32768 from the initial
    2048 across 7B to 65B LLaMA models using Position Interpolation. Our results show that

    1. Position Interpolation can easily enable very long context windows (e.g. 32768), requiring
    only fine-tuning for 1000 steps on the Pile (Gao et al., 2020) to achieve a good quality.
    The cost of fine-tuning is negligible compared to the pre-training costs. This confirms
    our hypothesis that it is relatively easy for the models to adapt to interpolated position
    encodings.

    2. Position Interpolation generates strong models that can effectively make use of much ex-
    tended context window. We show that models extended by Position Interpolation enjoy
    significant perplexity gains from greatly extended context windows for text modeling, and
    we show that the perplexity reduces graceful with the enlargement of context windows.
    We also applied Position Interpolation in a long text summarization task, and demonstrate
    competitive performances.

    3. Position Interpolation preserves model quality relatively well for tasks within its original
    context window sizes. We present a variety of evaluation results for the extended LLaMA
    models on the original LLaMA benchmark. Compared with original LLaMA models, the
    extended LLLaM A models saw a minor degradation on several standard benchmarks within
    a 2048 token limit.

    Our results highlight the innate ability of Transformer models to “extrapolate to sequence lengths
    longer than the ones encountered during training” as hypothesized in the seminal work of Vaswani
    et al. (2017). We reaffirm this hypothesis and suggest that the previously known weakness of ex-
    trapolating to longer sequences for language modeling (Press et al., 2022) may be due to direct

    extrapolation of positional encodings and it can be largely mitigated by interpolating position en-
    codings instead.

    Concurrent work. Right before our release, we are informed with a concurrent blogpost (Super-
    HOT kaiokendev (2023)) that also interpolates positional encoding in RoPE to extend the context
    window from 2K to 8K. Recently, open source community picks it up in Reddit post ! and Github
    Issues 2, which shows that fine-tuning with LoRA (Hu et al., 2021) also seems to work well. Our
    paper shows a full fine-tuning with up to 65B model work well with Position Interpolation, and we
    also give theoretical explanations why interpolation achieves much more stable results than extrap-
    olation, by showing that the upper bound of interplated attention score is much lower than that of
    extrapolated ones.

    Summary post for higher context sizes for this week. For context up to 4096, NTK RoPE scaling is pretty viable. For context higher than that, keep using SuperHOT LoRA/Merges.

    2 METHOD

    2.1 BACKGROUND: ROTARY POSITION EMBEDDING (ROPE)

    Transformer models require explicit positional information to be injected, typically in the form of
    positional encodings, to represent the order of inputs. We consider Rotary Position Embedding
    (ROPE) (Su et al., 2021), which is the position encoding used in the LLLaMA model (Touvron et al.,
    2023). Given a position index m € [0, ¢) and an embedding vector x := [zg, 71,..., 241], Where
    d is the dimension of the attention head, RoPE defines a vector-valued complex function f{x, m) as
    follows

    Using RoPE, the self-attention score
    is only dependent on relative position m — 7 through trigonometric functions. Here q and k are the
    query and key vector for a specific attention head. At each layer, RoPE is applied on both query and
    key embeddings for computing attention scores.

    2.2 DIRECT EXTRAPOLATION

    While the attention score in RoPE only depends on the relative positions, which is what we want,
    its extrapolation performance is not great . In particular, when directly extending to larger context
    windows unseen in the training, the perplexity may shoot up to very high numbers (i.e., > 10%),
    comparable to untrained models.

    Ideally, we want to see the model trained on a context window of size L = 2048 to still work
    reasonably well on longer context window, but may not have the capability to leverage information
    that appears beyond L. For example, to answer a question located at 3000, the model trained on
    maximal window size of I = 2048 cannot leverage evidences provided at location 0, but still
    can leverage the evidences provided at location 2900. In contrast, in reality we see catastrophic
    behaviors, i.e., question at location 3000 cannot be answered correctly, even if the evidences are
    located at location 2900.

    What is the reason behind? How could this happen if the attention score a,,,—,, decays as the relative
    distance |m — n/| increases, according to Section 3.4.3 of (Su et al., 2021), and content from very
    far distances should not matter that much? It turns out that the upper bound derived in Section 3.4.3
    of (Su et al., 2021) may be too loose: while it indeed decays with respect to |m — nl, the bound
    can still be quite large (i.e., the bound can be critically depends on the magnitude of v;) and thus
    vacuous. In fact, if we treat all trigonometric functions as basis functions (i.e, ¢;(s) := #93), and
    think about Eqn. 2 as basis expansion as the following:

    where s is the positional span between a query and a key and h; := (ga; + igaj+1){k2j — tk2j+1)
    are complex coefficients depending on q and k (here the definition of h; is exactly the same as the
    definition of k; in Sec 3.4.3 in RoPE (Su et al., 2021)). Now the the issue becomes clear: as shown
    in Fig. 2, a, can be small in magnitude in the range of [0, 2048], but gives huge values out of the
    region. The underlying reason is that the trigonometric family {¢;} (with sufficiently large d) is
    a universal approximator and can fit any arbitrary functions. Therefore, for a, there always exist
    coefficients {h;} (i.e. key and query) that corresponds to small function values in [0, 2048] but

    much larger in regions beyond.

    2.3 PROPOSED APPROACH: POSITION INTERPOLATION (PI)

    In Fig. 2, thanks to the smoothness of bases functions ¢; interpolation is much more stable and will
    not lead to wild values. Therefore, instead of extrapolate the attention score in Eqn. 3 to s > L,
    how about we define an attention score a{s) = a(Ls/L’) where L’ is the longer context window?
    Formally, we replace RoPE f by {’ defined as follows

    We call this transformation on the position encoding Position Interpolation. In this step, we reduce
    position indices from [0, L') to [0, L) to match the original range of indices before computing RoPE.
    Consequently, as inputs to RoPE, the maximum relative distance between any two tokens has been
    reduced from I’ to L. Since we align the ranges of position indices and relative distances before
    and after extension, we mitigate the effect on attention score computation due to context window
    extensions, which can allow the model easier to adapt. To further demonstrate this is the case, in the
    following theorem, we show that the interpolated attention score is well-behaved:

    While there is no close form for B(s) := 4/21 |Ag41(s)|, numerically it is at least larger than d, and for many positional difference s, B(s) is much larger than d
    (check Appendix B for the plot). Therefore, the interpolation bound is at least 2 - 294.73 ~ 600 x
    smaller than the extrapolation bound, and thus the interpolated attention score is much more stable
    than extrapolated one.

    Notably, our method of rescaling of position indices does not introduce extra weight, or modify
    the model architecture in any way. This makes it attractive in practical applications, since most
    infrastructure and optimization for the original model can be reused after the extension.

    Fine-tuning. We can further fine-tune the interpolated model using the next token prediction task
    with interpolated position encodings on the extended context window size using a pre-training cor-
    pus such as the Pile (Gao et al., 2020). In the next section, we show that our fine-tuning process
    only needs tens to hundreds thousands of examples. We also find that the result of the fine-tuning
    is not sensitive to the choice of examples. The reason may be that the model is only adapting to the
    new context window during the fine-tuning phase, starting from a good initialization, as opposed to
    acquiring new knowledge.

    Other ways to reduce interpolation/extrapolation bound. From the expression of the interpola-
    tion (Eqn. 5) and extrapolation bound (Eqn. 8), a common term is max; ||, which is the maximal
    magnitude of query/key products. If we enforce a regularization on || during LLM training, it is
    possible that the catastrophic extrapolation error can be mitigated or even resolved. In fact, if we
    apply ridge regression with proper regularization to fit a curve in Fig. 2, the magnitude of extrapo-
    lated a(s) when s > L can be comparable to that within [0, L]. To our knowledge, we are not aware
    of existing LLM pre-training techniques that leverage this regularization and will leave it for future
    work.

    3 EXPERIMENTS

    We show Position Interpolation can effectively extend context window up to 32 times of the original
    size, and such extension can be done with only several hundreds of training steps. We show the
    resulting models are strong LLMs with fully effective long context windows. We demonstrate its
    performance in a number of tasks including language modeling, passkey retrieval, and long doc-
    ument summarization. We also present benchmark results of the extended models on the original
    LLaMA evaluation benchmarks.
    3.1 SETUP

    Model Variants. We extended the pre-trained 7B, 13B, 33B and 65B LLaMA models (Touvron
    et al., 2023) to various context window of sizes up to 32768, using either direct fine-tuning or
    Position Interpoloation method. Except for rescaling the position indices for models extended with
    Position Interpolation, we did not modify LLaMA model architectures (Touvron et al., 2023) in any
    ways.

    Training Procedure. We fine-tune all model variants using the next token prediction objective. We
    use AdamW (Loshchilov & Hutter, 2019) with 5; = 0.9 and 2 = 0.95. We use a linear learning
    rate warmup of 20 steps starting from 10% of the maximum learning rate. For 7B and 13B models,
    we set the learning rate to 2 x 1075 and for 33B and 65B models we set the learning rate to 1072. We
    set the weight decay to zero. For extending 7B, 13B and 33B models to the 8192 context window
    size, we use 32 A100 GPUs and 64 global batch size. For all other cases we use 128 A100 GPUs and
    128 global batch size. We note that the main need of using more GPUs is memory limitation during
    fine-tuning, and it is possible to use fewer GPUs in certain cases. We train all models using PyTorch
    (Paszke et al., 2019) with Fully Sharded Data Parallel (Zhao et al., 2023) and Flash Attention (Dao
    et al., 2022).

    If not specified otherwise, for the Position Interpolation method, we fine-tune the models for 1000
    steps. For the direct fine-tuning method, we use 10000 steps. We primarily fine-tune using the Pile
    training dataset (Gao et al., 2020). In Section 3.4 we also compared fine-tuning performance on the
    RedPajama dataset (Computer, 2023).

    3.2 LONG SEQUENCE LANGUAGE MODELING

    We evaluate the long sequence language modeling performance of our extended models and base-
    lines on two datasets: book corpus (PG-19) (Rae et al., 2020) and cleaned Arxiv Math proof-pile
    dataset (Azerbayev et al., 2022).

    We use the test splits of PG19 (Rae et al., 2020) and proof-pile (Azerbayev et al., 2022). For PG19,
    we use the whole test split consisting of 100 documents. For the proof-pile dataset, we use a random
    subsample of 128 documents with at least 32768 SentencePiece (Kudo & Richardson, 2018) tokens
    and truncate to the first 32768 tokens for each test document. We evaluate perplexity at various
    context window size by using a sliding window approach following Press et al. (2022) with stride
    S = 256.

    In Table 1 and Table 2, we report the perplexity results for our models and baselines on the datasets.
    From the results, we found that models extended with our method enjoy a significantly improved
    perplexity from longer context window sizes. By increasing the context window size from 2048 to
    16384, we observed -0.28 and -0.5 reductions of perplexity for extending LLaMA 7B models on
    both datasets, -0.27 and -0.48 reductions for extending LL.aMA 13B models, and -0.14 and -0.42
    reductions for extending LLaMA 33B models. For LLaMA 65B models, we observed -0.12 and
    -0.3 reductions of perplexity by extending to the 8192 context window size.

    In general, we observed a consistent trend of our models achieving better perplexity with longer
    context windows. This indicates our models can effectively make use of the longer context windows
    to better predict next tokens in language modeling tasks. Moreover, we found this trend extends to
    32768 window size without diminishing on the PG19 dataset for LLaMA 7B and 13B models. This
    indicates that our method may enable extension to even longer context windows.

    In contrast, we observed that models extended via the direct fine-tuning method has shown regres-
    sion (up to +0.48) or minor improvement (up to -0.12) on the perplexity at longer context windows.
    This indicates that models extended this way have limited capability of making use of context win-
    dows longer than their pre-trained settings.

    We saw a minor degradation of the perplexity on the original context window of 2048 for our ex-
    tended models in some cases. For example, on the Proof-pile dataset, we saw a degradation ranging
    from 0.01 to 0.05 across all models with extended with Position Interpolation. A small degradation
    of performance within original evaluation context window is expected since Position Interpolation
    forces position encodings in original context window to reside in a much narrower region, which
    may negatively affect the language model’s performance. We present more benchmark results on
    the original context window size in Section 3.4.

    In Table 3 we report the relationship between perplexity and the number of fine-tuning steps for
    LLaMA 7B model extending to 8192 and 16384 context window sizes using Position Interpolation
    evaluated on the PG19 dataset. We can see without fine-tuning (at step 0) the model can exhibit
    certain language modeling capability, as indicated by < 20 perplexity for extending to 8192 context
    window (in contrast, the direct extrapolation method leads to > 10% perplexity). With fine-tuning,
    we observed that the perplexity improves quickly. At 200 steps the models surpassed the original
    model’s perplexity on 2048 context window size, indicating the models gaining ability of effectively
    using sequences longer than the pre-training settings for language modeling. At 1000 steps, we can
    see the models have improved steadily and achieve a significantly better perplexity.

    3.3 MEASURING EFFECTIVE CONTEXT WINDOW SIZE THROUGH PASSKEY RETRIEVAL

    We study the effective context window size, i.e. the maximum distance of a token can effectively
    attend to during inference, of our models after extension. To measure this, we follow a synthetic
    evaluation task of passkey retrieval proposed by Mohtashami & Jaggi (2023). In this task, the models
    are asked to recover a random passkey hidden in a long document. See Figure 3 for the format of
    the document.

    Given a language model, we estimate the upper and lower bounds of effective context windows as
    follows. Suppose the random passkey is k tokens away from the end of the input. When a model
    persistently fails to retrieve the correct passkey value across several independent attempts, it suggests
    that the effective context window size of the model is less than k. Conversely, if a model consistently
    succeeds in retrieving the correct passkey value, we deduce that the effective context window size
    of the model is at least k.

    We evaluate the 7B and 33B LLaMA model variants that are extended via Position Interpolation or
    direct fine-tuning. For each model, we use 32 different &£ uniformly spaced in the targeted context
    window L’ and run the above tests for 10 times for each k, where each time a random passkey of 5
    random digits is used. In Table 4, we report kyax as a function of the number of fine-tuning steps,

    We can see that models extended via Position Interpolation all successfully attain their desired ex-
    tension objectives in terms of effective context window sizes, indicating by the effective context
    window size reaching maximum kp, = L/, after merely fine-tuning for 200 steps, consistently
    across both 7B and 33B model sizes and up to 32768 context windows. In contrast, LLLaMA models
    that are extended via direct fine-tuning only saw a minimal increase of the effective context win-
    dow size kay from 2048 to 2560, even after fine-tuning for more than 10000 steps, with no clear
    indication of an acceleration in the increase of window size.

    3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE

    We evaluate the models extended by Position Interpolation on several standard benchmark tasks
    within the original context window size of 2048. The evaluation results are listed in Table 5. From
    the results, we saw that models extended to 8192 produce comparable results on the original bench-
    mark which is designed for a much smaller context window, with a degradation of up to 2% on
    the benchmark tasks, for both 7B and 33B model sizes. Models extended to longer context win-
    dows regressed more on the benchmarks, but still in reasonable ranges for most tasks. We also note
    that the choice of fine-tuning datasets does not seem to lead significant difference in the benchmark
    performances, which may be due to the limited number of fine-tuning steps used in our method.
    The regression on benchmark tasks is consistent with our observation on perplexity regression in
    Section 3.2.

    3.5 LONG DOCUMENT SUMMARIZATION

    In this task, we evaluate our models’ performance on the long document summarization task. In
    particular, we consider the GovReport (Huang et al., 2021) dataset, which contains 17457 documents
    for training and 972 documents for evaluation. Each document comes with a human generated
    summary. We truncate all input documents to their first 15000 tokens.

    We fine-tune the LL.aMA models extended with Position Interpolation with a context window of
    16384. Note the rescaling of position indices are still required during this fine-tuning step. We first
    Model Size Context Window Fine-tune on  BoolQ PIQA Race-M Race-H WinoGrande

    format the raw document using the prompt template in Figure 4, and then concatenate the prompt
    with the ground-truth summary (truncate to 1000 tokens) associated with each document. We fine-
    tune the model using the next token prediction task with the above setup for 10 epochs. The losses
    from the input prompt proportion of training examples are excluded during our fine-tuning.

    We use a generation temperature of 0.5 and top, = 0.95 as our inference parameter to generate a
    summarization of each document in the test set. The final output is truncated at 1000 tokens. We
    used the ROUGE-1/ROUGE-2/ROUGE-L scores (Lin, 2004) as the evaluation metrics to evaluate
    the models’ outputs vs the ground-truth summaries.

    In Table 6 we report our evaluation results. We have also included results from two baselines in
    existing SCROLLS Leaderboard (Shaham et al., 2022; Ainslie et al., 2023). In general, we have
    obtained competitive R1 score among other models with minimal tuning of hyper-parameters. This
    result suggests our models with 16384 context window can effectively handle the long document
    summarization task.

    === END OF FILE ===

    '''

    prompt_ext = '''
You are given this machine learning post, please read it carefully and answer the follow up question.

=== BEGIN ===
Hi there! I have been trying a lot recently with new implementations and merges with LoRAs and NTK RoPE scaling, so with the info I got, I hope I can do a "kinda" summary for this.

1 week ago or so, SuperHOT LoRAs got merged into a lot of models, managing to get pretty good results for contexts about 8K and 16K.

https://www.reddit.com/r/LocalLLaMA/comments/14kj2w8/thebloke_has_released_superhot_versions_of/

Then, some days ago, NTK RoPE scaling was discovered, which could in theory extend the context on base models without the need to finetune.

https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

Then, 2 days ago, it was discovered that Dynamic NTK RoPE scaling was possible, which let's you to adjust the alpha rope scaling dynamically based on context size.

https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/

Either NTK scaling method changes the rotatory embedding base value, while SuperHOT models change the RoPE value based on the Compression factor for positional embeddings.

So, after all this info, I can do a summary.

Based on the info of u/kaiokendev on his blog, https://kaiokendev.github.io/til#extending-context-to-8k, we can see that RoPE plus SuperHOT LoRA loses a bit of perplex vs base models, but it keeps getting a better perplex as you increase context.

That was wrong, check u/kaiokendev comment below.

Remember that for this, RoPE is set by the compress_pos_emb value.

PPL vs CTX with RoPE + SuperHOT LoRA

Now, on static NTK RoPE scaling, we see an issue past certain context values, and a really big penalty for bigger alphas.

Perplexity vs CTX, with Static NTK RoPE scaling

As can you see, NTK RoPE scaling seems to perform really well up to alpha 2, the same as 4096 context.

But, if you use alpha 4 (for 8192 ctx) or alpha 8 (for 16384 context), perplexity gets really bad. Alpha 4 starts to give bad resutls at just 6k context, and alpha 8 at 9k context. Both with a high perplex cost penalty at just smaller context sizes.

Then, dynamic NTK RoPE comes to the rescue, which you can see here.

Perplexity vs CTX, with Dynamic NTK RoPE scaling

Here, the dynamic alpha that changes based on the context size, keeps the perplexity on check until very high context sizes.

So at what point are we now?

SuperHOT LoRAs have been merged for a good amount of 13B and 30B models. 7B SuperHOT LoRA was released recently, and 65B SuperHOT LoRA is not out yet.

Static NTK RoPE scaling was added into exllama recently.

No implementation for now have been added for Dynamic NTK RoPE scaling (I've been trying on exllama, if you want to help check https://github.com/turboderp/exllama/issues/126) not possible at the moment.

And then, the summary goes like this at 2th July.

If you want to use 2k context, keep using base models.

If you want to use 4k context, Static NTK RoPE scaling with a value of 2 will yield you pretty good results. This is your only way for now for 65B models. You can also do it with SuperHOT LoRAs/Merges, but remember to use compression 4 for 8K models, and 8 for 16K models.

If you want to use 6k and higher context, use SuperHOT LoRA, or SuperHOT LoRAs merged with models. This is not feasible for now for 65B models.

After trying for like 5+ hours to implement Dynamic NTK RoPE scaling into exllama, I have to sleep (5AM)

Hope this post can help you guys on which models or technique to use for extended context sizes.

Just to add, so much have happened in just 1 week that my brain can't take more information anymore.

we can see that RoPE plus SuperHOT LoRA loses a bit of perplex vs base models, but it keeps getting a better perplex as you increase context

Ok, this is a complete misunderstanding of what linear interpolation is doing. Not saying you mean it intentionally, but the first portion here is wrong/misleading and this will only confuse people more.

The graph I posted there is from turboderp's original perplexity test comparing SuperHOT with compression factor of 4 to base LLaMA and one of his fine-tunes on 6K length data. It is only meant to illustrate that the perplexity decays as the sequence length grows longer -- it is using the longer context. It is not a proper comparison of the perplexity loss between scaling and base model. For that, you need to fine-tune your own model and run a perplexity test yourself. For example, here is my perplexity comparing base LLaMA, SuperHOT with interpolation, and SuperHOT with no interpolation against various scaling factors used during inference.

Comment Image
You can see the last line in this image is fine-tuned interpolated SuperHOT when the scaling factor matches the formula pretrained length / cutoff length and you can see it has the lowest perplexity and continues to decrease. I have echoed this several times in the last days: you do not lose perplexity when fine-tuning with linear interpolation, no matter the sequence length, as long as you use the correct scale. It is the same thing echoed in the Meta paper.

This also applies to the section "If you want to use 2K", if you are using a linear interpolated model that is fine-tuned on some sequence length, like SuperHOT 8K, and you are happy using it, there is no need to switch to a completely different model just to use a different sequence length as long as the implementation of it is correct. My heart goes out to turboderp as I'm sure he is still dealing with minor gotchas in the codebase that could have effects on the results, or for instance oobabooga as the exllama_hf loader had some problems. And then of course SuperHOT itself is just test at the end of the day, but the approach has already been validated.

Many of the confusion I have seen is a misunderstanding of what perplexity means, how to interpret the result, how to run the test, and what the test actually means for longer context cases (For instance, no tests used the sliding window evaluation, which is an even more important evaluation to run)

I don't mean to sound frustrated at you specifically, it is just a lot of compounding misinterpretations which is progressively making the discussion more confusing.

One other example is the dynamic scaling chart you posted. The only reason the dynamic has the same perplexity as the base model on <2048 length is because it **is** the base model -- no changes are being made to those lengths, it is the same thing if you had use the base model to begin with. The ppl increases after 2048 because it is not actually using the entire context. Yes, it doesn't explode as much, but it is still worsening the performance significantly. And yet I still see people saying they would like to use the dynamic version with no fine-tuning applied to it.

At this point, I can't keep on eye on all the interpretations or make sure no one misreads the implications of a result, and I can't look at every implementation in the codes and see if it is working the proper way or not. I think this is the last post I will make on the subject but I only can urge the community to understand how to interpret results posted and understand the metrics being used and what they mean (and what other metrics are out there that work better than just looking at the perplexity) I think there is a lot of value in NTK and whatever problems it has can be fixed with some more research and experimentation, but a post like this only makes it seem that the research is complete, and even presents a pseudo-ranking at the end based on misinterpretations. It is frustrating, but I know it will be better in time

I stand corrected, really thanks for your for your insight and correction of my post. A lot have happened in the last week, so I'm glad you're here as well to give the proper information.

I only know one thing about perplexity: low is good, high is bad. I believe most of the users in this subreddit are also in the same boat. I hope you don't feel discouraged by our ignorance. If there is a beginner's guide somewhere that explains the correct interpretation of perplexity, I wish someone could let me know where to find it.

The ppl increases after 2048 because it is not actually using the entire context. Yes, it doesn't explode as much, but it is still worsening the performance significantly.

I think we should stop using the sentence "it is not actually using the entire context", because it adds to the confusion. We should say instead: adding more context does not help in generation quality and perplexity. Even at alpha of 16 and beyond (with perplexity only increasing but not exploding), the network is able to use the entire context, it just that PPL increases so much that the network becomes incoherent. In a sense, PPL and context size is a tradeoff, if we can take the PPL hit for worse open-ended generation, we can take advantage of a much longer context size for information retrieval.

Note that this phenomenon is not just restricted to NTK-aware scaling, it is also seen for the orignal linear scaling method.

Also note that I am strictly talking about RoPE scaling on models that were never finetuned on RoPE scaling. Finetuning adds another layer of complexity that I am not ready to elaborate on. With finetuning, we can acheive better PPL and better long context retrieval, it is not a tradeoff anymore.

Here is a modified version of my original test, this time we truncate the prompt to the last ~1800 tokens instead of using RoPE scaling, and you can see that the answers are wrong or just complete nonsense (missing a lot of crucial information) compared to the original colab where I used the full prompt of 6200 tokens and alpha of 8.

https://colab.research.google.com/drive/1vedsFpScptZ46Fz4L3Dwd5T1pYwxeqgn

Edit: Original test with alpha of 8 for easy reference:

https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=d8aa5db3

According to the graphs we had, PPL without scaling at 1800 tokens is lower than alpha=8 at 6200, but clearely the high PPL alpha=8 answer is much more accurate than low PPL alpha=1 answer.

'''
    
    # Eval
    print("Full len:", len(prompt_full), "method:", meth)
    for i in range(100, word_len, 500): # 23700
        print("i:", i)
        prompt = prompt_full[:i]
        index = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        print("Prompt token length:", len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]))

        # print_predict(prompt, tokenizer)
        encoding = tokenizer(prompt, return_tensors="pt")
        input_ids = encoding["input_ids"].to("cuda")
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            ppl = torch.exp(outputs.loss)
            print("PPL:", ppl.item())
        
        pro_index[i//500] = index
        if args.method == "ntk":
            ppl_data[2, i//500] = ppl.item()
        elif args.method == "pi":
            ppl_data[1, i//500] = ppl.item()
        else:
            ppl_data[0, i//500] = ppl.item()
            


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="standard")
    parser.add_argument(
        "--scale", type=int, default=2
    )
    return parser.parse_args()

# import csv

# if __name__ == "__main__":
args = get_args() 
ppl_data = np.zeros((3, 50), dtype=np.float32)
pro_index = np.zeros(50, dtype=np.int32)
# standard, PI, NTK
word_len = 23700 # 23700
pro_len = 0
main(
    meth = args.method,
    scale = args.scale,
)

pro_index = pro_index[pro_index > 0]

store = np.vstack((pro_index, ppl_data[:, 0:len(pro_index)]))
          
name = args.method + "-" + str(args.scale) + ".csv"
np.savetxt(name, store, delimiter = ',')

# plt.xlabel("seq index")
# plt.ylabel("PPL")
# plt.plot(pro_index, ppl_data[2, 0:len(pro_index)], label="NTK")
# plt.plot(pro_index, ppl_data[1, 0:len(pro_index)], label="PI")
# plt.plot(pro_index, ppl_data[0, 0:len(pro_index)], label="Standard")
# plt.legend()


# plt.savefig('./test_ntk.jpg',  dpi=600)

