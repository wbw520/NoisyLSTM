import argparse


def get_arguments():
    """Parse all the arguments for model.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSP-Net Network")

    # train settings
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--sequence_len", type=int, default=4,
                        help="Length of sequence for LSTM model.")
    parser.add_argument("--data-dir", type=str, default="/home/wangbowen/PycharmProjects/city_data2/",
                        help="Path to the directory containing the image list.")
    parser.add_argument("--data-extra", type=str, default="/home/wangbowen/PycharmProjects/data_eye_train/",
                        help="Path to the directory of noise data")
    parser.add_argument("--original-size", type=int, default=[1024, 2048],
                        help="original size of data set image.")
    parser.add_argument("--need-size", type=int, default=[1024, 2048],
                        help="image size require for this program.")
    parser.add_argument("--input-size", type=int, default=[784, 784],
                        help="Comma-separated string with height and width of images. It also consider as crop size.")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--ignore-label", type=int, default=19,
                        help="this kind of pixel will not used for both train and evaluation")
    parser.add_argument("--num-classes", type=int, default=19,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epoch", type=int, default=40,
                        help="Number of training steps.")
    parser.add_argument("--seq", type=bool, default=False,
                        help="whether use LSTM model")
    parser.add_argument("--aux", type=bool, default=True,
                        help="whether use aux branch for training")

    # augment tools
    parser.add_argument("--random-crop", type=bool, default=True,
                        help="Whether to randomly crop the inputs during the training.")
    parser.add_argument("--random-mirror", type=bool, default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", type=bool, default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-rotate", type=bool, default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-sequence", type=bool, default=True,
                        help="Whether to random sequence.")
    parser.add_argument("--noise-ratio", type=int, default=50,
                        help="define the possibility of noise.")

    # evaluation settings
    parser.add_argument("--restore-from", type=str, default='./dataset/psp_pretrained_init.pth',
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default="PSP.pt",
                        help="name to save the model.")
    parser.add_argument("--gpu", type=str, default='cuda:0',
                        help="choose gpu device.")
    parser.add_argument("--multi", type=bool, default=True,
                        help="choose gpu device.")
    return parser.parse_args()


args = get_arguments()