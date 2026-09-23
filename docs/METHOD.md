# Method settings

## Pretraining

Dense ELECTRA replaced-token detection; generator and discriminator share word
embeddings. The generator has the same depth, one attention head, and one
quarter of the discriminator's hidden and feed-forward widths. Position tables
have 512 entries; pretraining uses sequences of 128 subwords.

The objective is generator cross-entropy plus 50 times discriminator binary
cross-entropy. Fifteen percent of eligible tokens are selected; 85% of selected
positions receive `[MASK]` at the generator input and the remainder retain their
input token. Generator samples use Gumbel-max. A sampled original token has RTD
label zero. Special tokens are not selected for corruption; discriminator loss
includes nonpadding tokens, as in the experiment implementation.

AdamW without bias correction uses beta = (0.9,0.999), epsilon = 1e-6, weight decay
0.01 except biases/LayerNorm, peak LR 5e-4, 10,000 warm-up updates and linear decay
to one million updates. Gradient norm is clipped to 1.0. Shared embeddings occur
only once in optimizer parameter groups. The first warm-up update has LR 0,
matching the archived LambdaLR schedule. Pretraining seed is 42.

## Punctuation fine-tuning

The final head is `Linear(hidden,512) -> SELU -> Linear(512,4)`. The default
`word_final` recipe removes punctuation before tokenization and applies weighted
cross-entropy to completed final subwords only. Padding and truncated partial
words do not contribute. Class weights in NONE/QUESTION/PERIOD/COMMA order are
1/5/2.5/1.5. The backbone LR is 5e-5; head LR 2e-4; batch 8, four epochs. AdamW uses
PyTorch's default beta/epsilon and weight decay 0.01. From epoch 2 onward, LRs are
multiplied by 0.95 every 10,000 within-epoch steps. Gradient norm is clipped to 1.0.
The final epoch is used; the validation set does not select a checkpoint.

Input sentences are lowercased, split with validation fraction 0.05 and split
seed 0, and assembled into blocks of 1-15 sentences. Trailing whitespace-separated
tokens are randomly removed using the archived block-construction routine.
Inputs are truncated at 512 subwords for fine-tuning. Fine-tuning seeds are
42,13,100. Evaluation uses complete word-final targets.

## Early exit

For L6 the exit depths are 2,3,4,5,6. Only added heads are trained, for two epochs
with batch 8, head LR 2e-4, weighted CE averaged over intermediate heads, and
no distillation term. Base/final-head parameters remain frozen; the archived
training procedure leaves backbone dropout enabled while training the heads.
Inference uses evaluation mode and float32. The first eligible head whose maximum
softmax probability is at least tau supplies the target label. All context tokens
are updated through that layer. The complete window stops at the target's exit.

The matched policy searches up to 1025 evenly indexed confidence midpoints, plus
0,1,0.99 and an explicit full-model endpoint. It minimizes average depth, subject
to the quality constraints described in README. Depth is not measured wall time.

## Selective KV reuse

Window 64 subwords; target word plus up to 4 future words are recomputed through all
layers. Previously processed left words reuse per-layer keys/values. The cache is
approximate because those representations were computed with older right context.
Absolute positions are anchored until the next index would exceed 127; then the
current window is recomputed from position 0 and the cache reinitialized.

Gate feature order for Small-L6 (268 values):

1. Four fast-branch class probabilities.
2. The target final-subword hidden state (256 values).
3. Eight scalars, in this exact order:
   - available future words /4;
   - current visible word length in subwords /8;
   - target offset within the window /63;
   - anchored target position /127;
   - reused prefix length /64;
   - window length /64;
   - indicator that the complete window was refreshed;
   - newly arrived subwords since the previous decision /8 (zero initially).

The MLP has 268 inputs, 32 ReLU units and one sigmoid output: 8,641 parameters.
Standardization uses fit-only means/stds, std floor 1e-4, then clipping to [-8,8].
Helpful-repair labels are `fast != gold and full == gold`. Weighted BCE uses
class weights 1/5/2.5/1.5 and a fit-only positive-class balance. Gate training uses
AdamW, LR 0.001, weight decay 0.001, 32 epochs, batch 1024 and gate seed 42.

Threshold selection minimizes repair fraction under calibration W-F1>=full and
per-punctuation F1>=full-0.5 points. Ties prefer higher W-F1. Candidate thresholds
are 1025 evenly indexed score midpoints plus always/never repair. Margin uses
1-(top1 probability-top2 probability); entropy uses natural logarithms. The gate
is trained independently for each punctuation checkpoint.

The repair path is a full forward pass over the same window with positions
starting from zero. It changes only the current label, never the cache. Combining
early exit with the repair branch is not the default method and is not exposed
as an equivalent implementation of the paper's main policy.
