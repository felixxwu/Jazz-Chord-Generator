import torch
import shutil
def saveCheckpoint(epoch, model, optimizer, is_best, name):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    f_path = f"output/{name}.pt"
    torch.save(state, f_path)
    if is_best:
        print('Best model so far, saving...')
        best_fpath = f"output/{name}-Best.pt"
        shutil.copyfile(f_path, best_fpath)

def loadCheckpoint(name, model, optimizer, best):
    f_path = f"output/{name}.pt"
    checkpoint = torch.load(f_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    best_fpath = f"output/{name}-Best.pt"
    bestState = torch.load(best_fpath)
    best.load_state_dict(bestState['state_dict'])

    return model, optimizer, checkpoint['epoch'], best