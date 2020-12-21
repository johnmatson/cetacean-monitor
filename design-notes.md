# Design Notes

## Server (Pi)
* Choose audio input during setup: disk or audio card
    * Audio on disk should be stored in large audio files

## Client (PC)
* Choose audio input during setup: disk or socket stream
    * Choose disk for training - data structured data with associated labels
    * Choose socket stream for testing - 