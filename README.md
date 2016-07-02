# CNNSaliencyMap
Given a pre-trained CNN, saliency map for an input image is generated corresponding to the output label of interest. The procedure followed is the one described in the paper ["Deep Inside Convolutional Networks"](https://arxiv.org/pdf/1312.6034.pdf). 

The saliency map generation is inspired by the basics of back propagation algorithm, which states that the deltas computed at a layer L equal the gradient of the loss incurred by the subgraph below L with respect to the outputs at L. Thus, backpropagating till the input data layer will yield us the gradient of the loss incurred by the whole CNN with respect to the input itself, thereby providing us the importance / saliency over the input image. 
