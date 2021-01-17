# CS294-158 (Spring 2019)

Deep unsupervised learning from @ucb

class video from https://www.bilibili.com/video/BV1Eb411Y7J5?p=1

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

  <img src="D:\data\WinHoliday_Sememster5\Notes\unsupervised learning\images\bayesnet.png" alt="image-20210116222119720" style="zoom: 150%;" />

- Autoregressive Models

  - 

- 

  

  



**3.Modern NN-based autoregressive models**





## Week 2









