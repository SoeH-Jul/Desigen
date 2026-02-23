import os
from skimage import io
import jittor as jt
from PIL import Image
from saliency.model import BASNet


def normPRED(d):
    ma = jt.max(d)
    mi = jt.min(d)

    dn = (d-mi)/(ma-mi)

    return dn


def save_output(image_name, pred, d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split("/")[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')


def get_saliency_model():
    print("...load BASNet...")
    cur_dir = os.path.dirname(__file__)
    model_dir = os.path.join(cur_dir, 'basnet.pth')
    net = BASNet(3, 1)
    net.load_state_dict(jt.load(model_dir))
    if jt.has_cuda:
        net.cuda()
    net.eval()

    return net


def saliency_detect(net, image, threshold=30):
    '''
    pred is ranged from 0 to 1
    '''
    with jt.no_grad():
        d1 = net(image)[0]
        pred = normPRED(d1[:, 0, :, :])
        if threshold is None:
            return pred
        return pred * 255 > threshold



    
