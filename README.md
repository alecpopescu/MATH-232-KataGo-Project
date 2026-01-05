# Dimension Reduction of Highly Dimensional Go Games

Group Final Project for MATH 2320: Advanced Linear Algebra with Applications.  
Authors, each with equal contribution: Alexander Popescu, Jake Todd, Haimanot Belachew, and Sara de Ángel

**Abstract:** In this project, our group attempted to create a binary classification neural network that predicts the outcome of Go games (black or white win) given a subset of moves from thousands of games. Given that Go is a highly complex game with nuanced strategy, we quickly found that, despite nontrivial improvement in classification accuracy, our model quickly overfit, indicating the need for more data and potentially a deeper network. To elucidate higher dimensional structure, we performed various linear and non-linear dimension reduction techniques, including principal component analysis, Laplacian eigenmaps, and diffusion maps. Laplacian eigenmaps and diffusion maps enabled us to identify potential trends that might explain differences in game trajectory based on whether white won or black won.

**Data Availability:** Our dataset was sourced from kata1 on the [KataGo platform](https://katagotraining.org/games/)
