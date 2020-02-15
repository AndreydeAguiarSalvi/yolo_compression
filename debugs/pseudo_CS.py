args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.which_gpu)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

icp_params = {'mask_initial_value': args.mask_initial_value}
train_loader, val_loader, test_loader = generate_loaders(args.val_set_size, args.batch_size, args.workers)

model = ResNet(args.mask_initial_value)

# After Train and Test function declarations
iters_per_reset = args.epochs-1
temp_increase = args.final_temp**(1./iters_per_reset)

trainable_params = filter(lambda p: p.requires_grad, model.parameters())
num_params = sum([p.numel() for p in trainable_params])
print("Total number of parameters: {}".format(num_params))

weight_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' not in p[0], model.named_parameters()))
mask_params = map(lambda a: a[1], filter(lambda p: p[1].requires_grad and 'mask' in p[0], model.named_parameters()))

model.ticket = False
weight_optim = optim.SGD(weight_params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.decay)
mask_optim = optim.SGD(mask_params, lr=args.lr, momentum=0.9, nesterov=True)
optimizers = [weight_optim, mask_optim]

for it:

	{ # Train Function
    for epoch:
        if epoch > 0: model.temp *= temp_increase # temperature is Beta in equations
        if it == 0 and epoch == args.rewing_epoch: model.checkpoint()
        for optimizer in optimizers: adjust_learning_rate(optimizer, epoch)

		for mini_batch:
            # After predictions
            masks = [m.masks for m in model.mask_modules]
            entries_sum = sum(m.sum() for m in masks)
            loss = Cross_Entropy + args.lmbda * entries_sum
            loss.backward()
            for optimizer in optimizers: optimizer.step()
        
        # Compute validations
        remaining_weights = compute_remaining_weights(masks)
    }
    model.temp = 1
    if it != rounds-1: model.prune() # it < 2

print('--------- Training final ticket -----------')
optimizers = [optim.SGD(weight_params, lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.decay)]
model.ticket = True
model.rewind_weights()
train(last_value_of_it+1)