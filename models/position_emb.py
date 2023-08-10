import torch
import os
from models.node2vec import Node2Vec




def position_emb(args, data=None, pe_method='node2vec'):
    if pe_method == 'node2vec':
        emb_path = os.path.join('dataset', args.dataset, args.node2vec_path,  str(args.node2vec_emb_dim)+'_pos_emb.pt')
        if args.node2vec_load:
            try:
                pos_emb = torch.load(emb_path)
                return pos_emb
            except:
                print('No pre-trained '+ pe_method +' embedding table!')
        else:
            model = Node2Vec(data.edge_index, args.node2vec_emb_dim, args.node2vec_walk_length,
                        args.node2vec_context_size, args.node2vec_walks_per_node,
                        sparse=True).to(args.device)
            loader = model.loader(batch_size=args.node2vec_batch_size, shuffle=True, 
                                  num_workers=4)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)
            model.train()
            
            for epoch in range(1, args.node2vec_epochs + 1):
                for i, (pos_rw, neg_rw) in enumerate(loader):
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(args.device), neg_rw.to(args.device))
                    loss.backward()
                    optimizer.step()

                    if (i + 1) % 10 == 0:
                        print(f'Epoch: {epoch:02d}, Step: {i+1:03d}/{len(loader)}, '
                        f'Loss: {loss:.4f}')

                    if (i + 1) % 1000 == 0:  # Save model every 100 steps.
                        # save_embedding(model)
                        if args.node2vec_save:
                            torch.save(model.embedding.weight.data.cpu(), emb_path)
                if args.node2vec_save:
                    torch.save(model.embedding.weight.data.cpu(), emb_path)
            if args.node2vec_save:
                torch.save(model.embedding.weight.data.cpu(), emb_path )
            # model = model.to('cpu')
            model = model.embedding.weight.data.cpu()
            # model = model.embedding.weight.data.cpu()
            return model
            