#!/usr/bin/python

""" given a number messages to/from POI (numerator)
    and number of all messages to/from a person (denominator),
    return the fraction of messages to/from that person
    that are from/to a POI
"""

import pickle
def computeFraction( poi_messages, all_messages ):

    if poi_messages == 'NaN':
        fraction = 0
    elif all_messages == 'NaN':
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)

    return fraction
