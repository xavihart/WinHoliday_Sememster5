CS294-158 (Spring 2020)

Deep unsupervised learning from @ucb

class video from(SP19) https://www.bilibili.com/video/BV1Eb411Y7J5?p=1

​                             (SP20)https://www.bilibili.com/video/BV1oE411F7iz?p=2

---

Notes by Haotian Xue from @sjtu

homepage : https://htxue.info :smile:

email : xavihart@sjtu.edu.cn :email:

---



## Week 1

**1.Motivation**

- likelihood-based models:
  estimate $p_{data}$ from samples $\{x^{(i)}\}$
- trade-off(to get the data distribution):
  - Efficient training and model representation
  - Expressiveness and generalization
  - Sampling quality and speed
  - Compression rate and speed

**2.Simple generative models**

- Just count it

  - JUST A histogram
  - fail in high dimension, poor in generalization
  - Solutions : function approximation  $p_{\theta}(x)$

- To get $p_{\theta}(x)$, maximum likelihood:
  $$
  argmin_{\theta}loss(\theta, x^{(1)},x^{(2)}, ...,x^{(n)} )=\frac{1}{n}\sum_{i=1}^{n}-log(p_{\theta}(x^{(i)}))
  $$
  等价于计算数据01分布和$p_{\theta}(x)$的KL散度最小

  -> Maximum likelihood + SGD

- (*) Bayes Network(Belif net / causal net)

  - DAG: vertex->property & edge->dependency & define parents and children
  - PGM(probability graph model) = Markov(无向) Net + Bayes Net(有向) 
  - sparsity the $2^i$ sized tabular

  <img src="images\bayesnet.png" alt="image-20210116222119720" style="zoom: 150%;" />

- Autoregressive Models

  - a fully expressive Bayes Net (just a chain rule model)

  - $logp(x)=\sum_{}^{}logp(x_i|x_{1:i-1})$

  - A toy example: p(x1, x2) = p(x1)p(x2|x1)

    - p(x1) : histogram
    - p(x2|x1): MLP with input x1 and output joint distribution of p(x2|x1)

    - Extent to high dimensions: 
      - only need O(d) Param instead of O(e^d) tabular Param
      - no share of information between different conditional distribution 

  - popular models:

    - RNN
    - Mask
      - masked MLP (MADE[masked auto encoder for distribution estimation])
        - satisfy the **autoregressive property**, the output of d dimension is only related to the input before the d dimension.
        - **more** to referred to in the MADE arxiv paper
      - masked convolutions 
        - use the convolutional kernel  <img src="images\conv.png" alt="image-20210116222119720" style="zoom: 50%;" />
        - limited receptive filed ; faster
        - Wave-Net <img src="images\wavenet.png" alt="image-20210116222119720" style="zoom: 50%;" /> dilated convolution
        - pixel-CNN (2016) :
          - combines two kind of convs together: vertical + horizontal
          - <img src="images\PIXELCNN.png" alt="image-20210116222119720" style="zoom: 50%;" />
        - gated pixel CNN :
          -  with improved conv structure : Gate Residual Block
      - self-attention 注意力机制

  


**3.Modern NN-based autoregressive models**



## Week 2

The most important idea for this lecture: 

*To fit the distribution for a dataset, instead of trying to get a direct output of the probability of inputs, we alternatively try to transfer the input space into a transformed space where the transformed samples fit some regular distributions such as the gaussian. After doing this, the sampling work will be much more easier and the output satisfy the $\sum p=1$.*

- Foundation of flows (1-D)

  - how to fit a density model

    - mixture of gaussians ?
      $$
      p_{\theta}(x) = \sum_{i=1}^{k}\pi_{i}\mathcal{N}(x;\mu_i;\sigma_i)
      $$

  not right for high dimensional data !

  <img src="images\flow.png" alt="image-20210116222119720" style="zoom: 90%;" />

  - x -> z , can calculate and get a bridge between p(x) and p(z)

  - After SGD optimization to get z, we sample z and project back to x to get the real sample 
  - 有点类似DIP里的直方图均衡

- 2-D flow ：the same as 1-D

  - autoregressive :

    - x1 -> z1 = f_1(x1)

    - x2 -> z2 = f_2(x1, x2)
      $$
      max_{\theta, \phi} = \sum \log p_{z_{1}}(f_{\theta}(x_1))+log|\frac{dz_1}{dx_1}|+\log  p_{z_{2}}(f_{\phi}(x_1, x_2)) + log|\frac{dz_2}{dx_2}|
      $$

- N-D flow 

  - Autoregressive flows and inverse autoregressive flow

    - Autoregressive flows(slow sampling + fast training)

      - $x_i \rightarrow z_i = f(x_{<i})$

      - training process:  $p_{\theta}(X) = p(f_{\theta}(X))|det\frac{\partial f_{\theta}(X)}{dX}| $  (Jacobian determinant)

        loss = -log(p(x)) = ...

      - sampling process (sample -> calculation)
        $$
        z_1\rightarrow x_1 = f^{-1}_{\theta}(z_1) \\
        z_2+x_1 \rightarrow x_2 =f^{-1}_{\theta}(z_2, x_1) \\
        z_3+x_1+x_2 \rightarrow x_3 =f^{-1}_{\theta}(z_3, x_1, x_2) \\
        $$
        it is actually serial -> very slow 

    - Inverse Autoregressive flows(slow training + fast sampling)
      $$
      z_1=f_{\theta}^{-1}(x_1) \\
      z_2=f_{\theta}^{-1}(x_2, z_1) \\
      z_3=f_{\theta}^{-1}(x_3, z_1, z_2) \\
      $$
      Then the sampling process becomes :
      $$
      x_1 = f_{\theta}(z_1) \\
      x_2 = f_{\theta}(z_1, z_2) \\
      x_3 = f_{\theta}(z_1, z_2, z_3) \\
      $$
      It can be paralleled !!

  - Flow decomposition, z1 = f1(f2(f3...fn(x)))))

    - problem : it is inefficient to calculate the Jacobi

  - Element-wise flows:

    - $f_{\theta}(x_1, x_2, ..., x_d) = (f(x_1), f(x_2), ...f(x_d))$
    - Jacobin mat is diagonal, so det() is easy to get 

  - RealNVP(NICE)-like arch [Dinh et al. Density estimation using Real NVP. ICLR 2017]

    - ![image-20210128171551503](.\images\RealNVP.png)

    - The Jacobin is a triangle mat, so the det value is easy to get.

    -  (补充) NICE arch and NVP arch

      -  **NICE: Non-linear Independent Components Estimation**
      - https://spaces.ac.cn/archives/5776 great learning material for NICE

      - NICE 结构，将原始数据x划分为两部分x1与x2，然后第一部分直接映射到z1而第二部分的映射z2=x2+m(x1)， 其中m是一个非线性的可逆可求导的函数。可以反复进行这样的耦合操作。 
        $$
        h_1 = x_1 \\
        h_2 = m(x_1) + x_2 \\
        $$

      - **NVP: Non-Volume Preserving transformations**

      - https://spaces.ac.cn/archives/5807/comment-page-1 great learning material
        $$
        h_1 = x_1 \\
        h_2 = s(x_1)*x_2 +t(x_2) \\
        $$

    ​           

  - Glow, Flow++, FFJORD

- dequantization :artificial_satellite:

  一种想法，把原来的离散数据加上扰动之后变成连续的数据

  

## Week 3 Latent Variable Models

Lot of math here

![image-20210128171551503](.\images\outline3.png)

#### Motivations

- Autoregressive-models + Flows VS latent variables:

  前者的变量都是可以观测的，后者有hidden variables

- $$
  z = (z_1, z_2, z_3, ...., z_K) = p(z, \beta) \\
  x = (x_1, x_2, x_3, ..., x_L) = p_{\theta}(x|z)
  $$

- For the above latent model

  - Sample : sample z from a easy distribution, then sample x
  - Evaluate: $p_{\theta}(x) = \sum_{z}p_Z(z)p_{\theta}(x|z)$
  - Train: $\max_{\theta}\sum_{i}log(p_{\theta}(x^{(i)}))$
  - Representation: 
  - (*) different from flow models:
    - the dimension of z and x may be different, there is no one to one map from z and x

  

#### Training latent variable models

- Objective : $\max_{\theta}\sum_{i}log(p_{\theta}(x^{(i)})) = \max_{\theta}\sum_{i}\sum_{z}(...)$ 

  - z can only take countable values : get all z
  - z can take impractically countable values : sample z

- Toy example : z  is a simple uniform distribution, p(x|z) is a gaussian determined by z and $\theta$. The raw distribution of x is a mixture gaussian. The results are as follows:![image-20210128171551503](.\images\latent_v_toy_exmaple.PNG)

- For z that cannot be counted easily: we can just sample z and then calculate the mean value of it 

- challenge: space of z is too big to be sampled correctly

  - We want to compute $E_{z\sim p_Z(z)}[f(z)]$
  - But where z is sampled most according to $p_Z$ may not have the main contribution to f(z), causing the expectation to be calculated inaccurately.

- Solution : important sampling for l v m:

  - Main idea: sample from another distribution q(z) instead of from p(z) .![image-20210128171551503](.\images\importance_sampling.PNG)

  - ![image-20210128171551503](.\images\sampling.PNG)

  - get q, find a gaussian q(z) to minimize KL(q(z), $p_{\theta}(x^{i}|z)$):

    ![image-20210128171551503](.\images\optim.PNG)

  - to make the above process faster, we use amortized formulation: ![image-20210128171551503](.\images\amortized.PNG)

    

    

    ![image-20210128171551503](.\images\amorti.PNG)

  - input x, output some important parameters for q(z) distribution such as $\mu , \sigma...$

  - The final objective become
    $$
    \sum_{i}\log \frac{1}{K}\sum_{k=1}^{K}\frac{z_{k}^{(i)}}{q(z_{k}^{(i)})}p_{\theta}(x^{(i)}|z_k^{(i)}) \\
    
    z_k^{(i) } \sim q(z_{k}^{(i)}) \\
    \min_{\phi}\sum_{i}KL(~q_{\phi}(z)~||~p_{\theta}(z|x^{(i)})~) \\
    \max(Term_1 - Term_2)
    $$
    ​	(IWAE : important weighted Auto Encoder)

  - Theorem1: more sampling will lower Loss(mathematically proved) 

#### Variations



#### Related Ideas



