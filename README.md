# Robot Frost

Generating poetry with recurrent neural networks.

## Usage

First, install the required dependencies:

```sh
pip3 install -r requirements.txt
```

#### Generating poems

Note that before you generate anything, you'll need a trained model.

If you try to generate a poem without a `./state.dict.pth` file, a new one will be made with random values, but this is not ideal.

You can download a trained model from the releases page, or see the next section for training your own.

Run the following command:

```sh
python3 robotfrost.py --generate True
```

You should see an A.I. generated poem!

You can set a poem length with `--poem-length <int>`, and some starting words to use with `--starting-words <str>`.

#### Training the model

Run the following command to start training:

```sh
python3 robotfrost.py --cuda True
```

If you don't have a compatible GPU with CUDA cores, remove `--cuda True`.

Press ^C (control c) to stop. You should see the new file `./state.dict.pth` which contains the trained model.

See some options you can play with:

```sh
python3 robotfrost.py --help
```

## Some poems by Robot Frost

```
Turns underscores
Mood
Come
Doctor? to laughed
Paradise-in-bloom, dream
Or every young house and house in all those frozen
It ought to know his how to keep well up
Being you might have name to bear a thing
That he was too long at the time for which I have to say,
You take a say
But in rain I should speak of it
Samoa,
Russia,
Ireland I complain and see for you,
Folks their
```
```
Lifelike path
Waiting
Steps is there? roads diverged in a figure in the shelves of all her at such a time?
It rests with the form
Of being anything
```
```
A flower bride, raspberries grow
Horse went outer must end kind, off light
Good-night, of following is books
But doubt are always
No matter for lantern-light long, yes, was,
May this a thing so very more best off a good one,
How they no one
To see the way he asked a thing like
Paul or be the thing so
```

## License

Every thing is licensed under the Unlicense (public domain equivalent).
