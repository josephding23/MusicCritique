import numpy as np
import pretty_midi


def get_onsets(path):
    pm = pretty_midi.PrettyMIDI(path)

    onsets = []
    for instr in pm.instruments:
        for note in instr.notes:
            onsets.append(note.start)
    return np.sort(onsets)


def estimate_tempi(path):

    # Grab the list of onsets
    onsets = get_onsets(path)
    # Compute inner-onset intervals
    ioi = np.diff(onsets)
    # "Rhythmic information is provided by IOIs in the range of
    # approximately 50ms to 2s (Handel, 1989)"
    ioi = ioi[ioi > .05]
    ioi = ioi[ioi < 2]
    # Normalize all iois into the range 30...300bpm
    for n in range(ioi.shape[0]):
        while ioi[n] < .2:
            ioi[n] *= 2
    # Array of inner onset interval cluster means
    clusters = np.array([])
    # Number of iois in each cluster
    cluster_counts = np.array([])
    for interval in ioi:
        # If this ioi falls within a cluster (threshold is 25ms)
        if (np.abs(clusters - interval) < .025).any():
            k = np.argmin(clusters - interval)
            # Update cluster mean
            clusters[k] = (cluster_counts[k]*clusters[k] +
                           interval)/(cluster_counts[k] + 1)
            # Update number of elements in cluster
            cluster_counts[k] += 1
        # No cluster is close, make a new one
        else:
            clusters = np.append(clusters, interval)
            cluster_counts = np.append(cluster_counts, 1.)
    # Sort the cluster list by count
    cluster_sort = np.argsort(cluster_counts)[::-1]
    clusters = clusters[cluster_sort]
    cluster_counts = cluster_counts[cluster_sort]
    # Normalize the cluster scores
    cluster_counts /= cluster_counts.sum()
    return 60./clusters, cluster_counts