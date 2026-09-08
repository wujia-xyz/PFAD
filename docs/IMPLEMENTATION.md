# PFAD implementation and configuration

These settings document the executed PFAD implementation and were moved out of the manuscript on 2026-09-06 at the author’s request. They are documentation, not a newly executed configuration.

Authoritative code: [core.py](../src/pfad/core.py). The active branch is the attentive-set student with soft geometric targets, record-wise rank selection, fixed depth correction and acquisition-topology meshing.

Sampling strides apply to image columns, not frames; all paired frames are processed. The source support and query candidates are separate sampled sets.

## Candidate and Evidence Construction

All paired frames were processed. Released response masks were binarized at intensities greater than or equal to 127 and skeletonized in two dimensions. We sampled image columns, using a stride of 8 with zero-based offset 0 for the dense source support and a stride of 32 with offset 4 for the query candidates. Both training and held-out queries followed the same rule. Every nonzero skeleton pixel in a sampled column was included, so a column could contribute more than one depth. Coordinates were computed at calibrated pixel centers, and response confidence was obtained by dividing the released intensity by 255.

Each query used up to $K=32$ neighbors from every other record. The distance offset was $\varepsilon_d=0.5$ mm, the ray--plane stability threshold was $\gamma_{\min}=0.15$, and the eigenvalue-ratio stabilizer was $\varepsilon_\lambda=10^{-8}$ mm$^2$. The CT teacher used anatomy temperature $\tau_a=1$ mm, surface tolerance $\rho=2$ mm, and surface temperature $\tau_s=0.25$ mm. These were fixed implementation settings. The numerical sigmoid used an exponent clipped to $[-40,40]$; this protects its evaluation rather than changing the stated distance tolerance.

## Feature Representation and Normalization

The source vector contains the eight entries in the record-evidence definition in the manuscript. Its displacement is divided by 2 mm and clipped to $[-4,4]$; the nearest and median distances are divided by 5 mm and clipped to $[0,8]$. The remaining five entries are clipped to $[0,1]$. Source sets are padded with zeros to the maximum available number of other records, with invalid entries masked in both pooling operations. This cohort requires at most five source entries per query.

The 19 query-context entries are ordered as confidence; depth, column, and frame fractions; three case-centered coordinates; three beam-direction components; median plane displacement; displacement disagreement; agreement; two median support distances; source count; and a three-component anatomy code. The code order is foot, tibia, and fibula. Query coordinates are centered by their case-wise median and divided by the 10th--90th percentile span on each axis, floored at 1 mm, before clipping to $[-3,3]$. For image height $H$, width $W$, pixel spacing $s$, and record length $F$, the depth denominator is $\max(Hs,10^{-6} \mathrm{mm})$, the column-center denominator is $\max(W,1)$, and the zero-based frame-index denominator is $\max(F-1,1)$.

The median displacement and disagreement use a 2-mm divisor and are clipped to $[-4,4]$ and $[0,4]$, respectively. Agreement is $\operatorname{clip}_{[0,1]}(1-\eta_i/s_\eta)$ with $s_\eta=1$ mm. Both median support distances use a 5-mm divisor and clipping to $[0,8]$; source count is divided by five. Beam directions, confidence, and anatomy indicators require no additional scaling. These transforms apply only to student inputs; the depth correction uses the unnormalized plane statistics.

## Student and Optimization

The shared source encoder has linear widths $8\rightarrow32\rightarrow32$, and the query encoder has widths $19\rightarrow32$. The fusion network has widths $83\rightarrow96\rightarrow48\rightarrow3$. Layer normalization follows the first linear layer of each encoder and the first fusion layer; GELU follows the hidden layers. Dropout of 0.08 is applied after the first fusion activation. Attention uses a bias-free linear map of the combined embeddings. The resulting student has 15,203 trainable parameters, with one model serving all three anatomies in each held-out fold.

Optimization used AdamW for 12 epochs, batches of 8,192 queries, learning rate $10^{-3}$, and weight decay $10^{-4}$. The PyTorch implementation used initialization seed 17 for the primary model. For a training set with $T$ queries and $C$ specimen--anatomy cases, the base weight of a query from a case containing $n_c$ queries was $T/(Cn_c)$. For each head, pooled hard-positive prevalence $\pi_h$ was clipped to $[10^{-3},1-10^{-3}]$. Positive and negative class factors were $0.5/\pi_h$ and $0.5/(1-\pi_h)$, respectively. Their products with the base weights were normalized to unit mean over the training set for each head. The loss was evaluated directly from logits for numerical stability.

## Reconstruction and Evaluation

Per-record selection used a minimum count $n_{\min}=3$, with ties retained. The geometric update used $B=2$ mm and $\beta=0.25$, giving a maximum displacement of 0.5 mm. Triangles were accepted only if their frame span was at most one, their column span at most 64 pixels, and their longest edge at most 5 mm. The triangle area-vector norm had to exceed $10^{-8}$ mm$^2$. Mesh cleanup removed degenerate triangles, duplicate triangles and vertices, and non-manifold edges. No additional outlier filter or closed-surface reconstruction was used for the primary acquisition-topology meshes.

For surface evaluation, we uniformly sampled 50,000 points from each mesh with sampling seed 260831. Point normals were estimated by principal component analysis over 24 nearest neighbors. Closest-surface queries and mesh operations used Open3D. These evaluation settings were shared across the compared point sets or surfaces as applicable.
