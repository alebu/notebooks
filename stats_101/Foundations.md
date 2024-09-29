# What you'll learn from this

* Samples and Statistics
* LLN
* Estimators
* CLT
* Sampling Distributions
* Confidence Intervals
* Hypothesis Testing


```python
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.stats import norm
palette = ['#001A72', '#FC4C02', '#41B6E6', '#B7C01B', '#00D4C5', '#00A499', '#DBE442']
bg = '#EBF6F3'
np.random.seed(42)
plt.rcParams.update({'axes.facecolor':bg})
plt.rcParams['axes.prop_cycle'] = cycler(color=palette)
plt.rcParams['figure.figsize'] = (10, 3)
plt.rcParams['axes.grid'] = True
```

## Probability Theory and Statistical Inference

**Probability theory** and **Statistical Inference** are complementary tools that we use to study aleatory behaviours. 

In Probability Theory, we use a mathematical framework to model behaviours we want to study, and then we use these models to deduce what is going to happen when we run experiments on them. This is a deductive process. 

In Statistical Inference, on the other hand, we observe the results of an experiment ran on the object of study and, after making some assumptions on its behaviour - in the form of a probability model built through Probability Theory - we analyse these results to learn something.

As an example, consider a coin as a the object of study. The landing of heads or tails when we toss the coin is an aleatory behaviour that we can model and study with these tools. We may assume that the probability of landing heads when tossed is 70% (the coin is not fair). Then, using probability theory, we can deduce other things: for example that the probability of landing tails is 30%, that the probability of landing two heads in a row in 49%, that the probability of landing 500 heads in a row is infinitesimaly small, and so on. This is a typical example of the deductive process conducted using probability theory.

On the other hand, suppose we are given a coin that we don't know and we want to learn whether it's fair or not. We might assume a few things about it: for example that the probability of getting heads when tossed doesn't change over time; that the result of each toss does not influence the others, and so on. Then we conduct an experiment, and toss it 100 times, getting 40 heads. What do we conclude? Is the coin rigged? This is the kind of question that statistical inference can help us answer.

### Logical Structure of Classic Statistical Inference

In classic (parametric) statistical inference, in order to learn something out of the data we get from an experiment, we normally follow this process:

* We assume an underlying probability model for the data-generating process; this model is built using Probability Theory, and has some **parameters** that we want to learn. This is essentially what we mean with *parametric statistics*: we encapsulate all the information we want to learn in some model parameters. In our coin example, we have one parameter: the probability of landing heads when we toss the coin.
* Hence, we consider the data coming from an experiment as random; again, in our coin example, if we toss the coin ten times, the number of heads is random.
* On the other hand, we consider the parameters unknown but deterministic and fixed.
* We use statistical inference to **estimate** the parameters we care about, hence learning something about our object of study.

As we already stressed enough, the key objective of this process is to learn some information about the object of study. We will always be **uncertain** about what we learn: we can never hope - if not in very trivial situations - to learn everything. Hence, as an output of this process, we don't only want an estimate our parameters, but also a quantification of our uncertainty about the estimates we get. However, because we are treating the model parameters as deterministic and fixed - but unknown - we cannot make probability statemets about them using this framework, which means we cannot quantify our uncertainty using probability distributions over these parameters. In our previous example, after tossing the coin n times and getting x heads, we cannot say something like "there is 50% probability that the coin is fair". This poses a problem: how can we quantify our uncertainty? The sleigh of hand used by classical statistical inference is to **make probability statements about the data**. We'll see in detail what this means.

## Probabilistic Foundations Of Statistical Inference

### Data, Random Variables and Statistics

As we said, in classic Statistical Inference we treat the data we observe as random. Hence, we usually model the it as a sequence of independent and identically distributed (from now on, i.i.d.) Random Variables, $X_1, X_2,  ..., X_n$. We will make some assumptions about the properties of these Random Variables, and usually try to learn something about them (the unknown parameters in our model of the data-generating process); one of the ways we can do this is by calculating a *statistic* $t = T(X_1, X_2,  ..., X_n)$, a function of the data that summarises some quantity or information useful for us. Note that the statistic is itself a random variable, being a function of random variables.  

An example of a statistic is the sample mean:

$$
    \bar{X} = \frac{X_1 + X_2 + ... + X_n}{n}
$$

In some lucky, particular cases, we can determine exactly the probabilistic behaviour of a statistic with a deductive process. For example, if $X_1, X_2,  ..., X_n$ are normally distribute with mean $\mu$ and variance $\sigma**2$, then:

$$
    \bar{X} = \frac{X_1 + X_2 + ... + X_n}{n} \sim \mathcal{N}(n\mu, n\sigma^2)
$$

That is, the sample mean is normally distributed, with mean $n\mu$ and variance $n\sigma^2$. This can be derived using the properties of normal random variables. Notice that this gives exactly what we need: a probability model of our data generating process: we know exactly how are sample mean is going to behave. This will allow us to quantify our uncertainty by making probability statements about the data - even we don't know yet how to do this.

However, we will not always be so lucky; we don't want to have to assume a model for our data only to make our calculations give us nice results. We want a way to generalise this method. For that, we will need a couple of theorems.

### The Strong Law of Large Numbers

Let $(X_i)_{i \in \mathbb{N}}$ be a sequence of i.i.d. Random Variables. Suppose that $\mathbb{E}[X_1] = \mu$ and $\mathbb{V}[X_1] = \sigma < \infty$. Then, with probability 1, $\bar{X} \rightarrow \mu$ as $n \rightarrow \infty$.

What this tells us is that if we take a sample that is big enough, we can use our statistic - the sample mean - to approximate (**estimate**) a parameter of our model - the expected value. Notice how we are making very little assumptions about the Random Variables that we are studying: they only need to have a finite Expected Value and Variance.

By carefully modelling our data, even by just using the sample mean, we can answer a variety of questions, including the one we considered in the introduction: it is sufficient to consider a random variable that takes the value 1 if the result of the coin is head, and 0 if it is tail. Then we set:

$$
    \mathbb{P}[X = 1] = p
$$

$$
    \mathbb{P}[X = 0] = 1 - p
$$

and we have:

$$
    \mathbb{E}[X] = p
$$

$$
\sigma^2 = \mathbb{V}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2 = p(1 - p) = .25
$$

and we are able to estimate the probability of the coin of landing on head using the sample mean. WE are able to achieve what we want - a probability model for the data - without making particularly strong assumptions. The Law of the Large numbers ensures us that with a big enough sample, we'll get to the right answer. Thanks to Python, we don't have to trust this. We can actually try it out. Using NumPy Random library, we can simulate a sample of 300 coin tosses from a fair coin:


```python
n = 300
possible_results = [0, 1]
mu = np.mean(possible_results)
sigma = np.sqrt(np.square(possible_results - mu).mean())
# Sampling
X_i = np.random.choice(possible_results, size = n)
```

We can then look at how the sample mean of the sample changes while n grows:


```python
plt.plot(X_i.cumsum()/np.arange(1, n + 1), label = r'$\bar{X}(n)$')
plt.axhline(np.mean(possible_results), color = palette[1], label = r'$\mu = $' + f'{mu}');
plt.legend();
plt.xlabel("n");
```


    
![png](Foundations_files/Foundations_6_0.png)
    


In the plot, we see exactly the kind of behaviour described by the SLLN: as n grows, the sample mean taken from the i.i.d. Random Variables converges to the expected value $\mu = 0.5$ (remember, we simulated a fair coin, that hence lands on head roughly half of the time). This is extremely important, as notice what we are saying: if we are given a sample that is big enough, we are guaranteed to learn something about the Random Variables we are studying, i.e., their expected value. As we are using Random Variables to model the world, this means we can learn something about the world through this procedure. This theorem lies at the centre of Statistical Inference.

When a statistic is used this way (we use it to estimate a parameter of our model), we call it an **Estimator**. By itself, the estimator only gives us a **Point Estimate** of the quantity we are trying to study, that is, our best guess about it. If the estimator is well built, it'll have some nice properties that ensure that our guess will really be the best guess given the information we have. However, it does not say anything about how certain or uncertain we are about the guess. 

In fact, correctly quantifying and communicating uncertainty is one of the main concerns of statistical inference. We want to be able to discern cases where we tossed a coin 10 times and got 4 heads from cases where we tossed it 10000 times and got 4000 heads. While the point estimates of p for these two cases would be the same, we intuitively understand that they would lead to drastically different conclusions about wheter the coin is fair or not. Hence, we need a way of encoding this uncertainty and manipulating it in our calculations.

In order to do that, the LLN is not enough: it only tells us that we'll learn something as n becomes very, very large. It does not tell us how large it needs to be to get to a certain precision in our estimates - or, in other words, how certain we are about our point estimates given a sample size n. For that, we need another theorem.

### The Central Limit Theorem

Using the setting from the theorem above, we also have:

$$
    X_1 + X_2 + ... + X_n \sim \mathcal{N}(n\mu, n\sigma^2)
$$

as $n \rightarrow \infty$. Technically, what we have in this theorem in a convergence in distribution. Without going into too much technical details, what this means is that with a large enough n, we can make approximate probability statements about the sum of the Random Variables.

Before looking at how this theorem is useful for statistical inference, we can notice that even just this first results gives us a lot of insight on one possible way normal distributions can arise. Whenever we have an outcome that is the result of a lot of individual random contributions being summed up, we tend to get a Normal distribution as a result. Let's check this result with Python. What we can do is:
* simulate a bunch (N = 1000) of samples of length n = 300, from the same random variables
* for each sample, calculate the sum
* analyse the histogram of the sums, and compare it to the Normal Distribution given by the theorem


```python
N = 1000
xs = np.random.choice(possible_results, size = (N, n))
```


```python
xs_sum = xs.cumsum(axis = 1)
sum_mean = n*mu
sum_sigma = np.sqrt(n)*sigma
rng = np.arange(sum_mean - 4*sum_sigma, sum_mean + 4*sum_sigma)
plt.plot(rng, norm.pdf(rng,sum_mean,sum_sigma), label = r'$\mathcal{N}(n\mu, \sigma\sqrt{n})$')
plt.hist(xs_sum[:, -1], bins = 50, density = True, label = r'$\sum{X_i}$' + " Histogram");
plt.legend()
plt.ylabel("Density");
```


    
![png](Foundations_files/Foundations_10_0.png)
    


In the plot, we have the histogram of our sample in orange, and the asymptotic distribution given by the CLT - $\mathcal{N}(n\mu, \sigma\sqrt{n})$ - in blue. We see that the histogram matches the distribution that we get from the theorem.

If we go back to our definition of the sample mean, we see that it involves a sum. Can we then use the CLT to infer something about the behaviour of the sample mean as n gets large? We can. By using some basic properties of Random Variables, we get:

$$
    \frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt n}} \sim \mathcal{N}(0, 1)
$$

In fact, this one, and not the previous one, is the precise statement of the CLT. Notice that this is exactly what we are after: a probabilistic description of how our random statistic behaves. We'll check this, computationally, by considering many samples (N = 1000) and looking at how their sample mean changes as n grows:


```python
xs_means = xs_sum/(np.arange(1, n + 1).reshape(1, n))
_, ax = plt.subplots()
for i in range(N):
    plt.plot(xs_means[i, :], color = palette[0], alpha = 0.01)
plt.plot(mu + 1.96*sigma/np.sqrt(np.arange(1, n + 1)), color = palette[1], label = r'$\mu \pm 1.96\frac{\sigma}{\sqrt{n}}$');
plt.plot(mu - 1.96*sigma/np.sqrt(np.arange(1, n + 1)), color = palette[1]);
plt.legend();
plt.xlabel("n");
```


    
![png](Foundations_files/Foundations_12_0.png)
    


We see that while the sample size grows, the variance of the sample means goes down. This matches our intuition: the bigger the sample, the more certain we will be about our estimate. Finally, we can see the distribution of the sample means for n = 300:


```python
mean_mean = mu
mean_sigma = sigma/np.sqrt(n)
rng = np.arange(mean_mean - 4*mean_sigma, mean_mean + 4*mean_sigma, 0.01)

plt.plot(rng, norm.pdf(rng,mean_mean,mean_sigma), label = r'$\mathcal{N}(\mu, \frac{\sigma}{\sqrt{n}})$')
plt.hist(xs_means[:, -1], bins = 50, density = True, label = r'$\bar{X}(n)$' + " Histogram");
plt.legend();
plt.ylabel("Density");
```


    
![png](Foundations_files/Foundations_14_0.png)
    


Again, in the plot we see the matching between the asymptotic distribution given by the CLT and the histogram of our sample. Think about what distribution is: it is the distribution of the Statistic we are studying, the sample mean. As we said before, being a function of random variables, it is a random variable itself. We'll call the distribution of a statistic the **Sampling Distribution**, and its standard deviation the **Standard Error**. 

Let's reflect on what we achieved:

* We discovered that if we have a large enough sample of i.i.d. random variables, the sample mean will be close enough to the expectation of the random variables.
* We also discovered what "close enough" means: the sampling distribution of the sample means is a Normal, centered around the mean mu, and with standard error inversely proportional to the square root of n and directly proportional to the standard deviation of the random variables we are studying.

The question, now, is how do we use these results for Statistical Inference. The typical situation we found ourselves in is to have a single sample (in our setting, N = 1), and from that we want to learn something about the data generating process. Thanks to the result above, after making some assumptions about the random variables we are studying, we can calculate the theoretical distribution of our statistic. How can we use this to quantify our uncertainty about the point estimates we calculate? In classical statistics, one of the simplest ways we have to do this is through the use of confidence intervals?

* Confidence Intervals
* Hypothesis Testing

## Confidence Intervals


```python
def calculate_95ci(X):
    sample_mean = X_i.mean()
    sample_variance = X_i.std()**2
    n = len(X)
    ci = (
        sample_mean - 1.96*np.sqrt(sample_variance/n),
        sample_mean + 1.96*np.sqrt(sample_variance/n)
    )
    return ci
```


```python
n = 30
X_i = np.random.choice(possible_results, size = 30)
calculate_95ci(X_i)
```




    (0.28814201332194833, 0.645191320011385)




```python
cis = []
sample_means = []
ci_contains_mu = []
m = 1000
for i in range(m):
    X_i = np.random.choice(possible_results, size = n)
    sample_mean = X_i.mean()
    sample_variance = X_i.std()**2
    ci = (
        sample_mean - 1.96*np.sqrt(sample_variance/n),
        sample_mean + 1.96*np.sqrt(sample_variance/n)
    )
    cis.append(ci)
    ci_contains_mu.append(
        (ci[0] <= .5)&
        (ci[1] >= .5)
    )   
    sample_means.append(sample_mean)


_, ax = plt.subplots()
ax.fill_between(
    np.arange(m),
    [ci[0] for ci in cis],
    [ci[1] for ci in cis]
)
ax.axhline(.5, color = palette[1])
```




    <matplotlib.lines.Line2D at 0x71f271339eb0>




    
![png](Foundations_files/Foundations_19_1.png)
    
