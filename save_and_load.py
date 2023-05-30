checkpoint_path = os.path.join(path, "checkpoint")

def save_ckp(state, is_best):
    f_path = os.path.join(checkpoint_path, 'checkpoint.pt') 
    torch.save(state, f_path)
    if is_best:
        best_fpath = os.path.join(checkpoint_path, 'best_model.pt') 
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath):
    checkpoint = torch.load(checkpoint_fpath)
    # model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint