---
layout: post
title: Second post
---


I tried my hands at processing images with fuel. When I tried to integrate it into a learning algorithm pipeline, more specifically that of a basic logistic regression model fit by gradient descent, I found that  obtaining  gradients and performing weight updates take little time compared to getting the data from a fuel datastream. This seems to be caused by the fact that the preprocessing (upscaling, cropping,flattening and rescaling) is more cpu intensive than the training step, therefore preprocessing the data paralleling does not resolve this issue. Perhaps this issue is caused by the inefficiency of my implementation; I will check other people's code to see if the same issue arises. 



In the process of experimenting with fuel.server and fuel.stream.ServerDataStream I encountered some (unresolved) technical issues which took me quite some time to get around. 

Below is one of the problems I encountered and the code that generates it.

[Error](https://github.com/yiulau/ift6266/blob/develop/nohup.out)

[Code that generates error](https://github.com/yiulau/ift6266/blob/develop/server.py)
