if __name__ == '__main__':
    try:
        import torch
    except ModuleNotFoundError:
        # This should be delegated to the installation script
        print('false')
    print(str(torch.cuda.is_available()).lower())
