def psnr(im, recon, verbose=True):
    im = np.squeeze(im)
    recon = np.squeeze(recon)
    MSE = np.sum((im - recon)**2) / (im.shape[0] * im.shape[1])
    MAX = np.max(im)
    PSNR = 10 * np.log10(MAX ** 2 / MSE)
    if verbose:
        print('PSNR %f'%PSNR)
    return PSNR

def plot_im(im, fname):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(2,2)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im, aspect='normal')
    fig.savefig(fname, dpi=50)

vars = scio.loadmat('inpaint_workspace')
re = vars['Dz']
re = re[5:105, 5:105,:]
orig = vars['b']

im_count = re.shape[-1]
fp = fopen('psnr.csv', 'w')
fp.write('IM No., CSC, ours \n')
for i in range(im_count):
    r = re[...,i]
    o = orig[...,i]
    plot_im(r, 'csc_re_%d'%i)
    plot_im(o, 'orig_%d'%i)
    pp = psnr(o, r)
    print('{}, {}, ''\n'.format(i, pp))

