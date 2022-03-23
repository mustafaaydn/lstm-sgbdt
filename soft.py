from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


class SoftInternalNode:
    def __init__(self, depth, tree, phi_numel=1):
        """
        An internal node of the soft-tree.

        Parameters
        ----------
        depth : int
            the depth of the node. "root" has depth 0 and it increases by 1.

        path_probability : float
            multiplication of probabilities (`self.prob`) of nodes that lead
            the way from the root to this node

        tree : SoftDecisionTree
            the tree that node belongs to

        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct). Passed to
            `make_children`.
        """
        self.depth = depth
        self.tree = tree

        self.is_leaf = False
        self.leaf_acc = []
        self.dense = nn.Linear(tree.input_dim, 1)
        # breakpoint()
        self.make_children(phi_numel=phi_numel)

    def make_children(self, phi_numel=1):
        """
        Every internal node is to have children and they are produced here.
        If at the penultimate depth, children will be leaves.
        """
        if self.depth + 1 == self.tree.depth:
            # We go 1 level deeper & reach the tree depth; they'll be leaves
            self.left = SoftLeafNode(self.tree, phi_numel=phi_numel)
            self.right = SoftLeafNode(self.tree, phi_numel=phi_numel)
        else:
            self.left = SoftInternalNode(self.depth + 1, self.tree,
                                         phi_numel=phi_numel)
            self.right = SoftInternalNode(self.depth + 1, self.tree,
                                          phi_numel=phi_numel)

    def forward(self, x):
        """
        Sigmoid softens the decision here.
        """
        return torch.sigmoid(self.dense(x))

    def calculate_probabilities(self, x, path_probability):
        """
        Produces the path probabilities of all nodes in the tree as well as
        the values sitting at the leaves. This is called only on the root node.
        """
        self.prob = self.forward(x)
        self.path_probability = path_probability
        left_leaf_acc = (self.left.calculate_probabilities(x,
                                                           path_probability
                                                           * (1-self.prob)))
        right_leaf_acc = (self.right.calculate_probabilities(x,
                                                             path_probability
                                                             * self.prob))
        self.leaf_acc.extend(left_leaf_acc)
        self.leaf_acc.extend(right_leaf_acc)
        return self.leaf_acc

    def reset(self):
        """
        Intermediate results i.e. leaf accumulations would cause "you're trying
        to backward the graph second time.." error so we "free" them here.
        """
        self.leaf_acc = []
        self.left.reset()
        self.right.reset()


class SoftLeafNode:
    def __init__(self, tree, phi_numel=1):
        """
        A leaf node of the soft-tree.

        Parameters
        ----------
        tree : SoftDecisionTree
            the tree that node belongs to
        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct).
        """
        self.tree = tree
        # breakpoint()
        self.phi = nn.Parameter(torch.randn(size=(phi_numel,)))
        self.is_leaf = True

    def calculate_probabilities(self, x, path_probability):
        """
        Since leaf, directly return its path_probability along with the value
        sitting at it.
        """
        return [[path_probability, self.phi]]

    def reset(self):
        """
        Keep the harmony
        """
        return


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, input_dim, phi_numel=1):
        """
        A soft binary decision tree; kind of a mix of what are described at
            1) https://www.cs.cornell.edu/~oirsoy/softtree.html and
            2) https://arxiv.org/abs/1711.09784

        Parameters
        ----------
        depth : int
            the depth of the tree. e.g. 1 means 2 leaf nodes.

        input_dim : int
            number of features in the incoming input vector. (needed because
            LazyLinear layer is not available yet in this version of PyTorch,
            which is 1.7)

        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct).
        """
        super().__init__()
        self.depth = depth
        self.input_dim = input_dim
        self.phi_numel = phi_numel
        # breakpoint()
        self.root = SoftInternalNode(depth=0, tree=self, phi_numel=phi_numel)
        self.collect_trainables()

    def collect_trainables(self):
        """
        Need to say PyTorch that we need gradients calculated with respect to
        the internal nodes' dense layers and leaf nodes' values. (since nodes
        are not an nn.Module subclass).
        """
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()
        while nodes:
            node = nodes.pop(0)
            if node.is_leaf:
                # breakpoint()
                self.param_list.append(node.phi)
            else:
                nodes.extend([node.left, node.right])
                self.module_list.append(node.dense)

    def forward(self, xs):
        """
        y_hat  = sum(pp_l * phi_l for l in leaf_nodes of self)
        """
        leaf_acc = self.root.calculate_probabilities(xs, path_probability=1)
        pred = torch.zeros(xs.shape[0], self.phi_numel)
        for path_probability, phi in leaf_acc:
            pred += phi * path_probability
        # don't forget to clean up the intermediate results!
        self.root.reset()
        return pred


class SoftGBM(nn.Module):
    def __init__(self, num_trees, tree_depth, input_dim, shrinkage_rate,
                 phi_numel=1):
        """
        Soft gradient boosting machine i.e. a GBM where base learners are soft
        binary decision trees

        Parameters
        ----------
        num_trees : int
            number of weak learners

        tree_depths : int or list of ints
            depth of each tree. If int, repeated. Else, must be a container
            of length `num_trees`.

        input_dim : int
            number of features in the input

        shrinkage_rate : float
            ~learning rate that determines the contribution of base learners.
            (not to be mixed with what the SGD-like optimizer will use)

        phi_numel : int, optional, default=1
            Number of elements residing in each leaf node's phi. If 1, one-step
            ahead forecasting is aimed, else multi-step (direct).
        """
        super().__init__()
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        self.input_dim = input_dim
        self.shrinkage_rate = shrinkage_rate
        self.phi_numel = phi_numel
        # breakpoint()
        self.trees = nn.ModuleList([SoftDecisionTree(depth=tree_depth,
                                                     input_dim=input_dim,
                                                     phi_numel=phi_numel)
                                    for _ in range(num_trees)])
        self.weak_predictions = []
        self.loss_fun = nn.MSELoss()
        self.batch_losses = []

    def forward(self, x):
        """
        Traverse through all the trees and ask their prediction; shrink it and
        add to
        """
        # breakpoint()
        self.weak_predictions = []
        overall_pred = torch.zeros(x.shape[0], self.phi_numel)
        for tree in self.trees:
            out = tree(x)
            overall_pred += self.shrinkage_rate * out
            self.weak_predictions.append(out)
        return overall_pred

    def get_loss(self, true):
        """
        Computes total loss i.e. sum of losses from each tree.
        """
        # the residuals are targets and it starts with y itself
        resid = true
        total_loss = torch.zeros(1)

        # for each tree..
        for j in range(self.num_trees):
            # get prediction of this tree
            pred_j = self.weak_predictions[j]
            shrunk_pred_j = self.shrinkage_rate * pred_j
            loss_j = self.loss_fun(shrunk_pred_j, resid)

            # update loss and residual
            total_loss += loss_j
            resid = resid - shrunk_pred_j

        return total_loss

    def train_(self, train_loader, optimizer, scheduler=None, num_epochs=10,
               print_every_e_epoch=1, print_every_b_batch=10,
               valid_loader=None, patience=None):
        # save to use later in forecasting
        self.train_loader = train_loader

        # train mode on..
        self.train()
        validate = valid_loader is not None
        for epoch in range(num_epochs):
            for batch_idx, (xb, yb) in enumerate(train_loader):
                # FiLiP ZuBU:
                _ = self(xb)                               # F-orward pass
                loss = self.get_loss(yb)                   # L-oss computation

                to_print = ((print_every_e_epoch is not None)
                            and (epoch % print_every_e_epoch == 0)
                            and (batch_idx % print_every_b_batch == 0))
                if to_print:                               # P-rint loss
                    print(f"Epoch {epoch}, batch {batch_idx}:"
                          f" loss = {loss.item()}")

                optimizer.zero_grad()                      # Z-ero gradients
                loss.backward()                            # B-ackward pass
                optimizer.step()                           # U-pdate parameters

                if validate:
                    pass
                self.batch_losses.append(loss)
            if scheduler is not None:
                scheduler.step()

    def predict(self, X):
        """
        One step ahead forward
        """
        return self.forward(X)

    @torch.no_grad()
    def predict_recursive(self, fh, new_index=None, true_values=None):
        if true_values is not None:
            true_values = torch.as_tensor(true_values, dtype=torch.float)

        # need last x & y values from training data
        xs, ys = self.train_loader.dataset.tensors
        last_xs, last_ys = xs[-1], ys[-1]

        preds = []
        # keep the lookback history in a fixed-size queue
        window_len = self.input_dim  # xs.shape[1]?
        fx = deque(torch.cat((last_xs[1:], last_ys)), maxlen=window_len)

        # for each horizon...
        for h in range(fh):
            # feedforward the current X
            fy, *_ = self(torch.as_tensor(fx).view(1, -1, 1))

            # right-append to queue the predicted value
            # and also store it
            fx.append(fy if true_values is None else true_values[h])
            preds.append(fy.item())

        return pd.Series(preds, index=new_index)

    @torch.no_grad()
    def predict_direct(self, new_index=None):
        # assumes model was trained with output window == forecast_horizon size
        # therefore it takes no `fh` parameter

        # need last x & y values from training data
        xs, ys = self.train_loader.dataset.tensors
        last_xs, last_ys = xs[-1], ys[-1]
        xy_numel_diff = last_xs.numel() - last_ys.numel()

        # take the "last" input from windows
        if xy_numel_diff >= 0:
            fx = torch.cat((last_xs[-xy_numel_diff:], last_ys))
        else:
            fx = last_ys[-xy_numel_diff:]

        # forward it
        fy, *_ = self(torch.as_tensor(fx).view(1, -1, 1))

        return pd.Series(fy, index=new_index)

    def predict_online(self, scaler, test_series, optimizer):
        # need last x & y values from training data
        xs, ys = self.train_loader.dataset.tensors
        last_xs, last_ys = xs[-1], ys[-1]

        # keep the lookback history in a fixed-size queue
        window_len = self.input_dim  # xs.shape[1]
        fx = deque(torch.cat((last_xs[1:], last_ys)), maxlen=window_len)

        # scale test values
        test_series = pd.Series(scaler.transform(
                                        test_series.to_numpy().reshape(-1, 1)
                                    ).ravel(),
                                index=test_series.index)

        # for each test value...
        preds = []
        for t_val in test_series.to_numpy():
            # predict with current X
            with torch.no_grad():
                fy, *_ = self(torch.as_tensor(fx, dtype=torch.float)
                                   .view(1, -1, 1))

            # `t_val` is revealed; update parameters
            new_tds = TensorDataset(
                            torch.as_tensor(
                                    np.array(fx).reshape(1, -1),
                                    dtype=torch.float),
                            torch.as_tensor(
                                    np.array(t_val).reshape(-1, 1),
                                    dtype=torch.float)
            )
            new_loader = DataLoader(new_tds, batch_size=1, shuffle=False)

            self.train_(new_loader, optimizer, scheduler=None,
                        num_epochs=1, print_every_e_epoch=1,
                        print_every_b_batch=1, valid_loader=None,
                        patience=None)

            # right-append to queue the *true* value
            fx.append(t_val)
            # store the prediction
            preds.append(fy.item())

        return pd.Series(preds, index=test_series.index)


class LSTSGBM(nn.Module):
    """
    LSTM + sGBDT
    """
    def __init__(self, lstm_hidden_size, lstm_input_size=1, lstm_num_layers=1,
                 lstm_dropout=0., lstm_forget_bias=1., hidden_pooling="last",
                 stateful=False, epoch_callback=None, **sgbm_kwargs):
        """
        Parameters
        ---------
        lstm_hidden_size : int
            Number of hidden units in an LSTM cell.

        lstm_input_size : int, optional, default=1
            Number of features of the input to feed in to an LSTM cell.

        lstm_num_layers : int, optional, default=1
            Number of LSTM layers to stack on top of each other.

        lstm_dropout : float, optional, default=0.
            Dropout ratio for the hidden units that is applied when passing
            from one LSTM layer to the next one; therefore only used iff
            `lstm_num_layers > 1`.

        lstm_forget_bias : float, optional, default=1.
            Following the advice in
            http://proceedings.mlr.press/v37/jozefowicz15.pdf, initalizes the
            forget bias of the LSTM with the given value (1. or 2. recommended)

        hidden_pooling : str, optional, default="last"
            How to pass the last hidden states of the LSTM to the sGBM. Options
            are "last" (to keep only the -1'th hidden state) or "mean" (to take
            the average of all the last layer hidden states).

        stateful : bool, optional, default=False
            Whether the underlying LSTM remembers the last state between
            batches.

        epoch_callback : callable, optional
            Will be called at the end of each epoch with parameters `epoch_no,
            train_loader, self`.

        **sgbm_kwargs
            Passed to the sGBM constructor. e.g: "num_trees": 2,
            "tree_depth": 1, "shrinkage_rate": 0.1, "phi_numel": 1.
        """
        super().__init__()
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.hidden_pooling = hidden_pooling
        self.stateful = stateful
        self.epoch_callback = epoch_callback
        self.sgbm_kwargs = sgbm_kwargs

        self.lstm = nn.LSTM(input_size=lstm_input_size,
                            hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers,
                            batch_first=True, dropout=lstm_dropout)
        if lstm_forget_bias is not None:
            self._init_forget_bias(self.lstm, fill_value=lstm_forget_bias)

        # provide some defaults (and fix `input_dim`)
        sgbm_kwargs = {"num_trees": 2, "tree_depth": 1, "shrinkage_rate": 0.1,
                       **sgbm_kwargs, "input_dim": lstm_hidden_size}
        # breakpoint()
        self.sgbm = SoftGBM(**sgbm_kwargs)

        # store training losses
        self.batch_losses = []

    def _init_forget_bias(self, lstm, fill_value=1.):
        """
        Adapted & modified:
        https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        """
        for coeff_names in lstm._all_weights:
            for bias_name in (coeff for coeff in coeff_names
                              if "bias" in coeff):
                bias = getattr(lstm, bias_name)
                n = bias.size(0)
                # [i | f | g | o]
                start, end = n//4, n//2
                bias.data[start:end].fill_(fill_value)

    def forward(self, x, init_lstm_state=None):
        # h_T is of shape: lstm_num_layers * num_directions, batch, hidden_size
        # hiddens is: batch, seq_len, num_directions * hidden_size
        hiddens, (h_T, c_T) = self.lstm(x, init_lstm_state)
        if self.hidden_pooling == "last":
            out = self.sgbm(hiddens[:, -1, :])
        elif self.hidden_pooling == "mean":
            out = self.sgbm(hiddens.mean(axis=1))
        return out, (h_T, c_T)

    def train_(self, train_loader, optimizer, scheduler=None, num_epochs=10,
               print_every_e_epoch=1, print_every_b_batch=10,
               valid_loader=None, patience=None):
        # save to use later in forecasting
        self.train_loader = train_loader

        try:
            self.sgbm.y_bar = train_loader.dataset.tensors[1].mean()
        except AttributeError:
            # "Subset" datasets have 1 more level of indirection
            self.sgbm.y_bar = train_loader.dataset.dataset.tensors[1].mean()

        # train mode on..
        self.train()
        validate = valid_loader is not None
        is_stateful = self.stateful
        for epoch in range(num_epochs):
            # hx is the 2-tuple of (h_0, c_0) initial states of LSTM
            # It becomes None (i.e, zeros) each time anew epoch begins
            # This way we aim for stateful LSTM
            hx = None
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb = xb.view(xb.size(0), -1, self.lstm_input_size)
                # FiLiP ZuBU:
                _, (h_T, c_T) = self(xb, init_lstm_state=hx)  # F-orward pass
                # if LSTM is stateful, remember last hidden state of the batch
                if is_stateful:
                    hx = (h_T, c_T)
                loss = self.sgbm.get_loss(yb)              # L-oss computation
                to_print = ((print_every_e_epoch is not None)
                            and (epoch % print_every_e_epoch == 0)
                            and (batch_idx % print_every_b_batch == 0))
                if to_print:                               # P-rint loss
                    print(f"Epoch {epoch}, batch {batch_idx}:"
                          f" loss = {loss.item()}")

                optimizer.zero_grad()                      # Z-ero gradients
                loss.backward()                            # B-ackward pass
                optimizer.step()                           # U-pdate parameters

                if validate:
                    pass
                # store loss
                self.batch_losses.append(loss)

            # an epch ends
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, train_loader, self)

            if scheduler is not None:
                scheduler.step()

    @torch.no_grad()
    def predict_recursive(self, fh, new_index=None, true_values=None):
        if true_values is not None:
            true_values = torch.as_tensor(true_values, dtype=torch.float)

        # need last x & y values from training data
        xs, ys = self.train_loader.dataset.tensors
        last_xs, last_ys = xs[-1], ys[-1]

        preds = []
        # keep the lookback history in a fixed-size queue
        window_len = self.sgbm.input_dim  # xs.shape[1]?
        fx = deque(torch.cat((last_xs[1:], last_ys)), maxlen=window_len)

        # for each horizon...
        for h in range(fh):
            # feedforward the current X
            fy, *_ = self(torch.as_tensor(fx).view(1, -1, 1))

            # right-append to queue the predicted value
            # and also store it
            fx.append(fy if true_values is None else true_values[h])
            preds.append(fy.item())

        return pd.Series(preds, index=new_index)

    @torch.no_grad()
    def predict_direct(self, new_index=None):
        # assumes model was trained with output window == forecast_horizon size
        # therefore it takes no `fh` parameter

        # need last x & y values from training data
        xs, ys = self.train_loader.dataset.tensors
        last_xs, last_ys = xs[-1], ys[-1]
        xy_numel_diff = last_xs.numel() - last_ys.numel()

        # take the "last" input from windows
        if xy_numel_diff >= 0:
            fx = torch.cat((last_xs[-xy_numel_diff:], last_ys))
        else:
            fx = last_ys[-xy_numel_diff:]

        # forward it
        fy, *_ = self(torch.as_tensor(fx).view(1, -1, 1))

        return pd.Series(fy.flatten(), index=new_index)

    def predict_online(self, scaler, test_series, optimizer,
                       print_every_e_epoch=1):
        # need last x & y values from training data
        xs, ys = self.train_loader.dataset.tensors
        last_xs, last_ys = xs[-1], ys[-1]

        # keep the lookback history in a fixed-size queue
        window_len = self.sgbm.input_dim  # xs.shape[1]
        fx = deque(torch.cat((last_xs[1:], last_ys)), maxlen=window_len)

        # scale test values
        test_series = pd.Series(scaler.transform(
                                        test_series.to_numpy().reshape(-1, 1)
                                    ).ravel(),
                                index=test_series.index)

        # for each test value...
        preds = []
        for t_val in test_series.to_numpy():
            # predict with current X
            with torch.no_grad():
                fy, *_ = self(torch.as_tensor(fx, dtype=torch.float)
                                   .view(1, -1, 1))

            # `t_val` is revealed; update parameters
            new_tds = TensorDataset(
                            torch.as_tensor(
                                    np.array(fx).reshape(1, -1),
                                    dtype=torch.float),
                            torch.as_tensor(
                                    np.array(t_val).reshape(-1, 1),
                                    dtype=torch.float)
            )
            new_loader = DataLoader(new_tds, batch_size=1, shuffle=False)

            self.train_(new_loader, optimizer, scheduler=None,
                        num_epochs=1, print_every_e_epoch=print_every_e_epoch,
                        print_every_b_batch=1, valid_loader=None,
                        patience=None)

            # right-append to queue the *true* value
            fx.append(t_val)
            # store the prediction
            preds.append(fy.item())

        return pd.Series(preds, index=test_series.index)
