---
layout: post
title:  "First post"
date:   2016-02-10 16:28:42 -0500
categories: jekyll update
---

I do not have a lof of experience with the more advanced OOP features of Python or with NUMPY, so I spent some time reading up on the documentation. Then I spent a lot of time trying to install theano on my computer and on clusters to which I have access. Since I do not have GPUs locally, I looked into SSH tunneling to interact with compute nodes on clusters using Jupyter notebook. This last attempt wasn't successful. A lot of time was also spent on setting up this blog using Jekyll. 

This is the first post. 

I have done some exploratory analysis of the dataset at hand. [This](https://github.com/yiulau/ift6266/blob/master/exploratory_analysis.ipynb)
is a link to the Jupyter notebook.

Below we test some markdown features.

Here is some python code:

```python
s = "Python syntax highlighting"
print s

>>> print(s)
Python syntax highlighting

```
Here is some R code:

```R
set.seed(3623293)
n<-100
ni<-rpois(n,4)+2
#Number of regressors
p<-1

> ni
  [1]  5  8  5  5 10  7  5  9  4  5  7  9  8  3  7  6  7  5  6  7  4  6 10  3
 [25]  7  4  4  2  4  5  6  4  3  7  4  7  6  7  5  6  5  5  7  5  7  6  6  7
 [49]  6  6  7  7  3  4  7  5  5  5  4  4  5  8  5  5  6  4  4  5  8  2  6  2
 [73]  6  3  6  5  6  8  5  5  7  9  7  5  8  4  5  8  9  6  6  5  8  6  4 10
 [97]  3  5  8  5
```

Include pictures/plots

![Picture of a cat]({{ site.baseurl }}/assets/2016-02-10/cat.jpg)


Formatting mathematical symbols with latex.


$$
\begin{align}
z=\sin(x)\\
\end{align}
$$

$$
\begin{align*}
z_1=\sin(x)\\
\end{align*}
$$

$$
\begin{align}
z_2=\sin(x)\\
\end{align}
$$

Matrices:

$$
\begin{align}
A= 
\begin{pmatrix}
1, &2 \\
2, &3 \\
\end{pmatrix}
\end{align}
$$

Some neural networks equations:
$$
\begin{align}
\mathbf{h} &=\sigma(W_1 \mathbf{x} +b_1) \\
\mathbf{z} &= g(W_2 \mathbf{h} +b_2) \\
\end{align}
$$

This is it for this post. In the next post I will explore Theano and train some neural networks. 


