import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def get_notes():
	notes = []
	for file in glob.glob("midi_songs/*.mid"):
		midi = converter.parse(file)
		print("Parsing %s" % file)
		to_parse = None
		try:
			s2 = instrument.partitionByInstrument(midi)
			to_parse = s2.parts[0].recurse()
		except:
			to_parse = midi.flat.notes
		for element in to_parse:
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))
	with open('data/notes', 'wb') as file:
		pickle.dump(notes, file)
	return notes

def prepare_sequences(notes, n_vocab):
	sequence_length = 100
	pitchnames = sorted(set(item for item in notes))
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
	network_input = []
	network_output = []
	for i in range(0, len(notes) - sequence_length, 1):
		seq_in = notes[i:i + sequence_length]
		seq_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in seq_in])
		network_output.append([note_to_int[seq_out]])
	n_patterns = len(network_input)
	network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
	network_input = network_input/float(n_vocab)
	network_output = np_utils.to_categorical(network_output)
	return (network_input, network_output)

def create_network(network_in, n_vocab):
	model = Sequential()
	model.add(LSTM(512, input_shape=(network_in.shape[1], network_in.shape[2]), recurrent_dropout=0.3, return_sequences=True))
	model.add(LSTM(512, recurrent_dropout=0.3, return_sequences=True))
	model.add(LSTM(512))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(n_vocab))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
	return model

def train(model, network_input, network_output):
	file = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
	checkpoint = ModelCheckpoint(file, monitor='loss', verbose=0, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

def train_network():
	notes = get_notes()
	n_vocab = len(set(notes))
	net_input, net_output = prepare_sequences(notes, n_vocab)
	model = create_network(net_input, n_vocab)
	train(model, net_input, net_output)

if __name__ == '__main__':
	train_network()