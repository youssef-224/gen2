from music21 import *
import keras
from keras import backend as k
import glob
import numpy
from keras.utils import np_utils 
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,LSTM
from keras.optimizers import RMSprop
from flask import Flask, jsonify, request, send_file
from music21 import *

print("choese categore")
print("hiphop                 piano")
print("pop                    j-pop")
print("pokemon           beethoven")
print("chopin       ")
#x= input()
print("Waiting time: 3 minutes")

app = Flask(__name__)

@app.route('/generate_music', methods=['GET'])
def generate_music():
    x = request.args.get('x', '')
#-------------------hiphop----------------------------------    
    if x == "hiphop":
        notes = []
        counter = 0
        for file in glob.glob("hiphopnew/*.mid"):
            
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))


        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)
        
        #------------------------------------------------------------------------------ 
        #------------------------------------------------------------------------------         
        #model = Sequential()
        #model.add(LSTM(
        #       256,
        #        input_shape=(network_input.shape[1], network_input.shape[2]),
        #        return_sequences=True
        #    ))
        #model.add(Dropout(0.3))
        #model.add(LSTM(512, return_sequences=True))
        #model.add(Dropout(0.3))
        #model.add(LSTM(256))
        #model.add(Dense(256))
        #model.add(Dropout(0.3))
        #model.add(Dense(n_vocab))
        #model.add(Activation('softmax'))
        #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        #model.summary()

        #filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    
        #checkpoint = keras.callbacks.ModelCheckpoint(
        #    filepath, monitor='loss', 
        #    verbose=0,        
        #    save_best_only=True,        
        #    mode='min'
        #)    
        #callbacks_list = [checkpoint]     
        #model.fit(network_input, network_output, epochs=200, batch_size=512,verbose=1, callbacks=callbacks_list)
        #-------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
           512,
          input_shape=(network_input.shape[1], network_input.shape[2]),
         return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        ## Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-hiphop.hdf5')
        
        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
            
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='hiphop_test_output.mid')  
        with open('hiphop_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': 'hiphop_test_output.mid'}
                    return jsonify(response)


    #-------------------piano----------------------------------            
    elif x==  "piano"  :
        notes = []
        counter = 0
        for file in glob.glob("piano/*.mid"):
            
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    

        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-piano.hdf5')

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='piano_test_output.mid')    
        with open('piano_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': 'piano_test_output.mid'}
                    return jsonify(response)

    #-------------------pop----------------------------------             
    elif x==  "pop"  :
        notes = []
        counter = 0
        for file in glob.glob("pop/*.mid"):
            
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))


        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-pop.hdf5')

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='pop_test_output.mid')  
        with open('pop_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': 'pop_test_output.mid'}
                    return jsonify(response)


    #-------------------j-pop----------------------------------           
    elif x==  "j-pop"  :
        notes = []
        counter = 0
        for file in glob.glob("j-pop/*.mid"):
            
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))


        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)
    

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-j-pop.hdf5')

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp=' j-pop_test_output.mid')               
        with open(' j-pop_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': ' j-pop_test_output.mid'}
                    return jsonify(response)

    #-------------------beethoven----------------------------------            
    elif x==  "beethoven"  :
        notes = []
        counter = 0
        for file in glob.glob("Beethoven/*.mid"):
            
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))


        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-Beethoven.hdf5')

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='beethoven_test_output.mid')     
        with open('beethoven_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': 'beethoven_test_output.mid'}
                    return jsonify(response)


#-------------------chopin----------------------------------             
    elif x==  "chopin"  :
        notes = []
        counter = 0
        for file in glob.glob("chopin/*.mid"):
        
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))


        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-chopin.hdf5')

        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='chopin_test_output.mid')  
        with open('chopin_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': 'chopin_test_output.mid'}
                    return jsonify(response)



    #-------------------pokemon----------------------------------           
    elif x==  "pokemon"  :
        notes = []
        counter = 0
        for file in glob.glob("Pokemon MIDIs/*.mid"):
            
            counter = counter + 1
            midi = converter.parse(file)
            notes_to_parse = None
            parts = instrument.partitionByInstrument(midi)
            if parts: # file has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # file has notes in a flat structure
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))


        sequence_length = 100
        n_vocab = len(set(notes))

        # get all pitch names
        pitchnames = sorted(set(item for item in notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
        network_input = []
        network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(notes) - sequence_length, 1):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]
            network_input.append([note_to_int[char] for char in sequence_in])
            network_output.append(note_to_int[sequence_out])
        n_patterns = len(network_input)
        # reshape the input into a format compatible with LSTM layers
        network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
        # normalize input
        network_input = network_input / float(n_vocab) 
        network_output = np_utils.to_categorical(network_output)
    

        # now we will generate the music
        model = Sequential()
        model.add(LSTM(
            512,
            input_shape=(network_input.shape[1], network_input.shape[2]),
            return_sequences=True
        ))
        model.add(Dropout(0.3))
        model.add(LSTM(512, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(512))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Dense(n_vocab))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        # Load the weights to each node
        #model.load_weights('/content/drive/MyDrive/generation -20230430T160244Z-001/generation/weights-simple piano.hdf5')

        
        start = numpy.random.randint(0, len(network_input)-1)
        int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
        pattern = network_input[start]

        prediction_output = []
        # generate 500 notes
        for note_index in range(500):
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            #index = numpy.argmax(prediction)
            index=numpy.random.choice(len(prediction.flatten()), 1, replace=True, p=prediction.flatten())
            index= index[0]
            #print(index)
            result = int_to_note[index]
            prediction_output.append(result)
            ind =numpy.asarray([index])
            pattern = numpy.append(pattern, ind)
            pattern = pattern[1:len(pattern)]

        offset = 0
        output_notes = []
            # create note and chord objects based on the values generated by the model
        for pattern in prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            # increase offset each iteration so that notes do not stack
            offset += 0.5

        midi_stream = stream.Stream(output_notes)
        midi_stream.write('midi', fp='pokemon_test_output.mid')   
        with open('pokemon_test_output.mid', 'rb') as f:
                    midi_data = f.read()
                    response = {'midi_data': midi_data.hex(), 'filename': 'pokemon_test_output.mid'}
                    return jsonify(response)

    else:
        # Invalid value for x
        return 'Invalid value for x', 400 
if __name__ == '__main__':
    app.run(debug=True)                    