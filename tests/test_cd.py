import numpy as np
import torch
import sys
import acd
import pickle as pkl
import warnings
warnings.filterwarnings("ignore")

    
def test_mnist(device='cuda'):
    # load the dataset
    sys.path.append('../')
    import dsets
#     sys.path.append('../dsets/mnist')
#     sys.path.append('../../dsets/mnist')
#     import dsets.mnist.model
    from dsets.mnist.model import Net
#     device = 'cuda'
    im_torch = torch.randn(1, 1, 28, 28).to(device)

    # load the model
    model = Net().to(device)
    model.load_state_dict(torch.load('../dsets/mnist/mnist.model', map_location=device))
    model = model.eval()
    
    # check that full image mask = prediction
    preds = model.logits(im_torch).cpu().detach().numpy()
    cd_score, irrel_scores = acd.cd(im_torch, model, mask=np.ones((1, 1, 28, 28)), model_type='mnist', device=device)
    cd_score = cd_score.cpu().detach().numpy()
    irrel_scores = irrel_scores.cpu().detach().numpy()
    assert(np.allclose(cd_score, preds, atol=1e-2))
    assert(np.allclose(irrel_scores, irrel_scores * 0, atol=1e-2))

    # check that rel + irrel = prediction for another subset
    # preds = preds - model.hidden_to_label.bias.detach().numpy()
    mask = np.zeros((28, 28))
    mask[:14] = 1
    cd_score, irrel_scores = acd.cd(im_torch, model, mask=mask, model_type='mnist', device=device)
    cd_score = cd_score.cpu().detach().numpy()
    irrel_scores = irrel_scores.cpu().detach().numpy()
    assert(np.allclose(cd_score + irrel_scores, preds, atol=1e-2))

    
if __name__ == '__main__':
    print('testing mnist...')
    test_mnist()

    print('all passed!')
    # loop over device types?

    # try without torch.no_grad()?