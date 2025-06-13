from mammoth import train, load_runner, get_avail_args

def main():
    required_args, optional_args = get_avail_args(dataset='seq-cifar10', model='sgd')

    print("Required arguments:")
    for arg, info in required_args.items():
        print(f"  {arg}: {info['description']}")

    print("\nOptional arguments:")
    for arg, info in optional_args.items():
        print(f"  {arg}: {info['default']} - {info['description']}")

    model, dataset = load_runner('sgd','seq-cifar10',{'lr': 0.1, 'n_epochs': 1, 'batch_size': 32})

    train(model, dataset)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()