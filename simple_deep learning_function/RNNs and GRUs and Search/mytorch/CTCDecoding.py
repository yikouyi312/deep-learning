import numpy as np


class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """
        self.symbol_set = symbol_set

    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1

        # TODO:
        # 1. Iterate over sequence length - len(y_probs[0])
        for t in range(0, len(y_probs[0])):
            # 2. Iterate over symbol probabilities
            # 3. update path probability, by multiplying with the current max probability
            # 4. Select most probable symbol and append to decoded_path
            index = np.argmax(y_probs[:, t, 0])
            # if index != blank and index != pre_index:
            #     index = np.argsort(y_probs[:, t, 0])[-2]
            path_prob *= y_probs[index, t, 0]
            if index != blank:
                if len(decoded_path) == 0 or decoded_path[-1] != self.symbol_set[index - 1]:
                    decoded_path.append(self.symbol_set[index - 1])
        # 5. Compress sequence (Inside or outside the loop)
        decoded_path = ''.join(decoded_path)
        return decoded_path, path_prob
        # raise NotImplementedError


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
        self.blank = 0

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, 0
        # First time instant: Initialize paths with each of the symbols,
        # including blank, using score at time t=1
        for bath_iter in range(0, y_probs.shape[2]):
            NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore \
                = self.InitializePaths(y_probs[:, 0, bath_iter])
            # Subsequent time steps
            for t in range(1, T):
                # Prune the collection down to the BeamWidth
                PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = \
                    self.Prune(
                        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore,
                    )
                # First extend paths by a blank
                NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank,
                                                                                    PathsWithTerminalSymbol,
                                                                                    BlankPathScore,
                                                                                    PathScore,
                                                                                    y_probs[:, t, bath_iter])
                # Next extend paths by a symbol
                NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank,
                                                                                 PathsWithTerminalSymbol,
                                                                                 BlankPathScore,
                                                                                 PathScore,
                                                                                 y_probs[:, t, bath_iter])
            MergedPaths, FinalPathScore_per = \
                self.MergeIdenticalPaths(NewPathsWithTerminalBlank,
                                         NewBlankPathScore,
                                         NewPathsWithTerminalSymbol,
                                         NewPathScore)

            for path in MergedPaths:
                if FinalPathScore_per[path] > FinalPathScore:
                    FinalPathScore = FinalPathScore_per[path]
                    bestPath = path

        return bestPath, FinalPathScore_per

    def InitializePaths(self, y_prob):
        InitialBlankPathScore = {}
        InitialPathScore = {}
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = '-'
        InitialBlankPathScore[path] = y_prob[self.blank]  # Score of blank at t=1
        InitialPathsWithFinalBlank = [path]
        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = []
        # This is the entire symbol set, without the blank
        for index, c in enumerate(self.symbol_set):
            path = c
            InitialPathScore[path] = y_prob[index + 1]  # Score of symbol c at t=1
            InitialPathsWithFinalSymbol += path  # Set addition
        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        scorelist = []
        # First gather all the relevant scores
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])
        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist = sorted(scorelist, reverse=True)  # In decreasing order
        if len(scorelist) < self.beam_width:
            cutoff = scorelist[-1]
        else:
            cutoff = scorelist[self.beam_width - 1]
        PrunedPathsWithTerminalBlank = []
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.append(p)  # Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]
        PrunedPathsWithTerminalSymbol = []
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.append(p)  # Set addition
                PrunedPathScore[p] = PathScore[p]
        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, y_prob):
        UpdatedPathsWithTerminalBlank = []
        UpdatedBlankPathScore = {}
        # First work on paths with terminal blanks
        # (This represents transitions along horizontal trellis edges for blanks)
        for path in PathsWithTerminalBlank:
            # Repeating a blank doesn’t change the symbol sequence
            UpdatedPathsWithTerminalBlank.append(path) # Set addition
            UpdatedBlankPathScore[path] = BlankPathScore[path] * y_prob[self.blank]
        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank# simply add the score.
            # If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += PathScore[path] * y_prob[self.blank]
            else:
                UpdatedPathsWithTerminalBlank.append(path)  # Set addition
                UpdatedBlankPathScore[path] = PathScore[path] * y_prob[self.blank]
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, y_prob):
        UpdatedPathsWithTerminalSymbol = []
        UpdatedPathScore = {}
        # First extend the paths terminating in blanks. This will always create a new sequencefor path in PathsWithTerminalBlank:
        for path in PathsWithTerminalBlank:
            for index, c in enumerate(self.symbol_set):  # SymbolSet does not include blanks
                if path[-1] == '-':
                    newpath = c
                else:
                    newpath = path + c  # Concatenation
                UpdatedPathsWithTerminalSymbol.append(newpath)  # Set addition
                UpdatedPathScore[newpath] = BlankPathScore[path] * y_prob[index + 1]
        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
            for index, c in enumerate(self.symbol_set):  # SymbolSet does not include blanks
                if c == path[-1]:
                    newpath = path
                else:
                    newpath = path + c
                # Horizontal transitions don’t extend the sequence
                if newpath in UpdatedPathsWithTerminalSymbol:
                    # Already in list, merge paths
                    UpdatedPathScore[newpath] += PathScore[path] * y_prob[index + 1]
                else:
                    # Create new path
                    UpdatedPathsWithTerminalSymbol.append(newpath)  # Set addition
                    UpdatedPathScore[newpath] = PathScore[path] * y_prob[index + 1]
        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore
        # Paths with terminal blanks will contribute scores to existing identical paths from
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        blank = False
        for p in PathsWithTerminalBlank:
            if p != '-':
                if p in MergedPaths:
                    FinalPathScore[p] += BlankPathScore[p]
                else:
                    MergedPaths.append(p)  # Set addition
                    FinalPathScore[p] = BlankPathScore[p]
            else:
                MergedPaths.append('')
                FinalPathScore[''] = BlankPathScore[p]
        FinalPathScore = dict(sorted(FinalPathScore.items(), key=lambda item: item[1], reverse=True))

        return MergedPaths, FinalPathScore
