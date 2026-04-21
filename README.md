#### GLASS Flows Tutorial

GLASS Flows are an inference-timer wrapper around pre-trained flow models converting sampling from an "ODE" paradigm to an "SDE"-like paradigm in which particle paths are refreshed by "inner" glass flows. Paper: https://arxiv.org/pdf/2509.25170.

##### Experiments

The notebooks build on each other in this order:

glass_flows_tutorial.ipynb — Start here. Introduces the GlassFlow class on a synthetic 2D checkerboard dataset with a from-scratch flow model. Covers the theory (Algorithm 1), stochastic transitions, and comparison to SDEs.

glass_flows_klein_exp.ipynb — First real-image experiment. Applies Glass Flows to the Flux.2-klein-base-4B model; establishes the image sampling setup used by all subsequent notebooks.

reward_guidance_klein.ipynb — Adds reward guidance: differentiable HPSv2 reward gradients are injected into the Glass Flow velocity at each inner step. Introduces GlassFlowBar_X_s (gradient on bar_X_s) and GlassFlowBar_X_sWeighted (gradient reweighted by σ²/α). 

fks_steering_glass_flows_klein.ipynb — Feynman-Kac Steering (SMC): particles evolve through Glass Flow transitions, resampled at each backbone step by HPSv2 score. Proposal is the plain GlassFlow with no reward gradient.

fks_and_reward_guidance_klein.ipynb — Combines both: FKS loop using GlassFlowBar_X_sWeighted as the proposal, so reward guidance shapes each transition and SMC selection pressure is applied at each backbone step.


##### Environment Requirements
Listing the base model and reward models here. 

pip install git+https://github.com/black-forest-labs/flux2.git \
pip install hpsv2 \
pip install imscore

##### Device

Note: This code is written for Apple Silicon (MPS). To run on CUDA, replace device="mps" with device="cuda" throughout, and update.
