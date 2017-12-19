import sys
import time

def show_time_elapsed(start_time):
    elapsed_time = time.time() - start_time

    print(str(elapsed_time))


def show_progress(iter, total):

    print("iteration " + str(iter) + "/" + str(total))