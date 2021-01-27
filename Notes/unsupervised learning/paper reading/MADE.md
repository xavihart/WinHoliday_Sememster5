**MADE: Masked Autoencodr for Distribution Estimation**

arXiv:1502.03509 [cs, stat] version2015-06-05

---

- Background 
  - 使用NN来估计样本分布（estimate p(x) from {X^(t)}）的工作：
    - 一般我们使用类似Negative log likelihood 的loss，但是存在问题：没有概率意义
      - $p(x) = \Pi_{i=1}^{d}\hat{x}_d^{x_d}(1-\hat{x}_d)^{1-x_d}$, 如果这样定义p(x)的话，假设训练拟合到最好，那么对于任意的输入x，我们可以拟合出一个完美的Auto-Encoder使$\hat{x}$始终等于$x$.导致$p(x)=1 ~ if ~ x\in X$, so $\sum_{x}p(x)\neq 1$.
  
- Methology

  - Using widely-adapted chain rule, so we have $p(x) = \Pi [p(x_d|x_{<d})]$

  - Get a valid log likelihood : 
    $$
    -log(p(x)) = \sum_{d=1}^{D}-logp(x_d|x_{<d})
    $$

  - where we can find that the value of  p(x|x<d) must only be dependent with the value before x, also called the autoregressive property.
  - Based on the above property we can design the network structure (略，比较直观)
  - order-agnostic training , 每一次更新batch之后，改变特征在chain rule中的顺序

- Experiment 

  - NLL performance on test set 
  - image generation

- Some questions or toughts