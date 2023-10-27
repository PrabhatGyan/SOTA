# State-of-the-art Seminar

## **Title** : Generative AI : An Overview

### **_By_** : Gyan Prabhat
### **Supervisor**: [Dr. Anand Mishra](https://anandmishra22.github.io/) <br>

#### Overview
We are in the era of Generative Artificial Intelligence (AI). With the paradigm shift of just looking for the patterns in the data, we have taken a step forward where we want to generate brand new data instances based on the patterns we have learnt from the given data. Deep Generative Modelling can be considered as a particular subset of Deep Learning where the end objective is to generate brand-new data instances. These are typically unsupervised learning methods, where we are provided with the training data X which is coming from a probability distribution $p_{data}(x)$. However, we do not know whatsoever about the data-generating distribution. What we want to learn is a $p_{model}(x)$, such that the learn distribution is as close as possible to the original data distribution. If we can do that, then we can use the learned distribution to sample new data instances.

### **Resources**
- Short Presentation - [Link]()
- Detailed Report - [Link]()

### Tutorial Plan
1. **Introduction**
   * Machine Learning Paradigm
     - Supervised vs Unsupervised Models
   * Discriminative Models
   * Generative Models
   * Why Generative Models?
       - Debiasing
       - Outlier detection
2. **Autoencoders**
   * Introduction
   * Latent Space
   * Architecture
     - Loss Function
   * Applications
     - Feature Learning
     - Dimensionality Reduction
   * Takeaways
   * Generation
3. **Variational Autoencoders**
   * Introduction
   * Traditional vs Variational Autoencoders
   * Architecture
     - Encoder
     - Decoder
   * Loss Function
     - Reconstruction Loss
     - Regularization Loss
4. **Generative adversarial networks (GAN)**
   * Introduction
     - Issues
     - Ideas
   * Architecture
     - Generator Network
     - Discriminator Network
   * Training
     - Min-Max Game
     - Gradient Updates
5. **Diffusion Models**
   * Introduction
     - What is Diffusion?
     - Diffusion in AI
     - Trailer of Diffusion model
     - Intuition of Diffusion model
   * Architecture
     - Diffusion Process
       - Forward Noising
       - Reverse Denoising
     - Sampling brand new Generations
6. **Evaluation of Generative Models**
   * Know the Ground Truth
     - MSE (Mean Squared Error)
     - PSNR (Peak Signal to Noise Ratio)
     - SSIM (Structural Similarity)
   * Unknown Ground Truth
     - IS (Inception Score)
     - FID (Fréchet Inception Distance)
     - KID (Kernel Inception Distance)
   * Human Evaluation
  7. **Hands-on**
     * Autoencoders
     * GAN

### Reference Papers
1. Bank et al., [Autoencoders](https://arxiv.org/pdf/2003.05991.pdf)
2. Kingma et al., [Auto-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114v10.pdf)
3. Sohl-Dickstein et al., [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)
4. Song et al., [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/pdf/1907.05600.pdf)
5. Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf) [Github](https://github.com/hojonathanho/diffusion)


[Contact : Gyan Prabhat](prabhat.1@iitj.ac.in)


