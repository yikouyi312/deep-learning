import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """
        extended_symbols = [self.BLANK]
        for symbol in target:
            extended_symbols.append(symbol)
            extended_symbols.append(self.BLANK)
        N = len(extended_symbols)
        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        skip_connect = [0, 0]
        pre = extended_symbols[1]
        for i in range(2, N):
            if extended_symbols[i] != pre and extended_symbols[i] != self.BLANK:
                skip_connect.append(1)
                pre = extended_symbols[i]
            else:
                skip_connect.append(0)
        extended_symbols = np.array(extended_symbols).reshape((N,))
        skip_connect = np.array(skip_connect).reshape((N,))

        return extended_symbols, skip_connect
        # raise NotImplementedError

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros(shape=(T, S))
        # -------------------------------------------->
        # TODO: Intialize alpha[0][0]
        # TODO: Intialize alpha[0][1]
        # TODO: Compute all values for alpha[t][sym] where 1 <= t < T and 1 <= sym < S (assuming zero-indexing)
        # IMP: Remember to check for skipConnect when calculating alpha
        # Consider s,
        # if skip_connect == 1
        # previous step can be BLANK(extended_symbols[s-1]), extended_symbols[s] and extended_symbols[s-2]
        # else
        # previous step can be BLANK(extended_symbols[s-1]), extended_symbols[s]
        # <---------------------------------------------
        # Initialization
        alpha[0][0] = logits[0, self.BLANK]
        alpha[0][1] = logits[0, extended_symbols[1]]
        # Iterate
        for t in range(1, T):
            alpha[t][0] = alpha[t-1][0] * logits[t, self.BLANK]
            for s in range(1, S):
                alpha[t][s] += (alpha[t-1][s] + alpha[t-1][s-1]) * logits[t, extended_symbols[s]]
                if skip_connect[s] == 1:
                    alpha[t][s] += (alpha[t-1][s-2]) * logits[t, extended_symbols[s]]

        return alpha
        raise NotImplementedError

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """
        S, T = len(extended_symbols), len(logits)
        beta = np.zeros(shape=(T, S))
        betahat = np.zeros(shape=(T, S))
        # -------------------------------------------->
        # TODO
        # Consider s,
        # if skip_connect == 1
        # next step can be BLANK(extended_symbols[s+1]), extended_symbols[s],
        # and for s-2, there is extra term extended_symbols[s]
        # else
        # next step can be BLANK(extended_symbols[s+1]), extended_symbols[s]
        # <--------------------------------------------
        # Initialization
        betahat[T-1][S-1] = logits[T-1, extended_symbols[S-1]]
        betahat[T-1][S-2] = logits[T-1, extended_symbols[S-2]]
        # Iterate for T-2 to 0
        for t in range(T-2, -1, -1):
            betahat[t][S-1] += betahat[t+1][S-1] * logits[t, extended_symbols[S-1]]
            for s in range(S-2, -1, -1):
                betahat[t][s] += (betahat[t+1][s] + betahat[t+1][s+1]) * logits[t, extended_symbols[s]]
                if skip_connect[s] == 1:
                    betahat[t][s-2] += betahat[t+1][s] * logits[t, extended_symbols[s-2]]
        #derive beta from betahat
        for t in range(T-1, -1, -1):
            for s in range(S - 1, -1, -1):
                beta[t][s] = betahat[t][s] / logits[t, extended_symbols[s]]
        return beta
        raise NotImplementedError

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """
        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        gamma = alpha * beta
        sumgamma = np.expand_dims(np.sum(gamma, axis=1), axis=1)
        gamma = gamma / sumgamma
        return gamma
        raise NotImplementedError


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

		Initialize instance variables

        Argument(s)
		-----------
		BLANK (int, optional): blank label index. Default 0.

		"""
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

		Computes the CTC Loss by calculating forward, backward, and
		posterior proabilites, and then calculating the avg. loss between
		targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses will be divided by the target lengths
        #####  and then the mean over the batch is taken

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Loss for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute forward probabilities
            #     Compute backward probabilities
            #     Compute posteriors using total probability function
            #     Compute expected divergence for each batch and store it in totalLoss
            #     Take an average over all batches and return final result
            # <---------------------------------------------
            #     Truncate the target to target length
            target = self.target[batch_itr][:target_lengths[batch_itr]]
            #     Truncate the logits to input length
            logits = self.logits[:input_lengths[batch_itr], batch_itr, :]
            #     Extend target sequence with blank
            extended_symbols, skip_connect = self.ctc.extend_target_with_blank(target)
            self.extended_symbols.append(extended_symbols)
            #     Compute forward probabilities
            alpha = self.ctc.get_forward_probs(logits, extended_symbols, skip_connect)
            #     Compute backward probabilities
            beta = self.ctc.get_backward_probs(logits, extended_symbols, skip_connect)
            #     Compute posteriors using total probability function
            gamma = self.ctc.get_posterior_probs(alpha, beta)
            self.gammas.append(gamma)
            #     Compute expected divergence for each batch and store it in totalLoss
                #  Compute logits
            logits = logits[:, extended_symbols]
            total_loss[batch_itr] = - np.sum(gamma * np.log(logits))
        #     Take an average over all batches and return final result
        total_loss = np.sum(total_loss) / B

        return total_loss
        raise NotImplementedError

    def backward(self):
        """

		CTC loss backard

        Calculate the gradients w.r.t the parameters and return the derivative
		w.r.t the inputs, xt and ht, to the cell.

        Input
        -----
        logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
			log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
            derivative of divergence w.r.t the input symbols at each time

        """

        # No need to modify
        T, B, C = self.logits.shape
        dY = np.full_like(self.logits, 0)

        for batch_itr in range(B):
            # -------------------------------------------->
            # Computing CTC Derivative for single batch
            # Process:
            #     Truncate the target to target length
            #     Truncate the logits to input length
            #     Extend target sequence with blank
            #     Compute derivative of divergence and store them in dY
            # <---------------------------------------------
            #     Truncate the target to target length
            target = self.target[batch_itr][: self.target_lengths[batch_itr]]
            #     Truncate the logits to input length
            logits = self.logits[: self.input_lengths[batch_itr], batch_itr, :]
            #     Extend target sequence with blank
            extended_symbols = self.extended_symbols[batch_itr]
            #     get gamma matrix
            gamma = self.gammas[batch_itr]
            #     Compute derivative of divergence and store them in dY
            #      dY,  logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
            T, S = gamma.shape
            for t in range(T):
                for s in range(S):
                    dY[t, batch_itr, extended_symbols[s]] -= gamma[t][s] / logits[t, extended_symbols[s]]
            # -------------------------------------------->
            # TODO
            # <---------------------------------------------
            pass

        return dY
        raise NotImplementedError
