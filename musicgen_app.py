# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import subprocess as sp
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import Softmax

import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen, MultiBandDiffusion


MODEL = None  # Last used model
IS_BATCHED = "facebook/MusicGen" in os.environ.get('SPACE_ID', '')
print(IS_BATCHED)
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break


file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/musicgen-melody'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        MODEL = MusicGen.get_pretrained(version)


def load_diffusion():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, melodies, duration, progress=False, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts, [None if m is None else (m[0], m[1].shape) for m in melodies])
    be = time.time()
    processed_melodies = []
    target_sr = 32000
    target_ac = 1
    for melody in melodies:
        if melody is None:
            processed_melodies.append(None)
        else:
            sr, melody = melody[0], torch.from_numpy(melody[1]).to(MODEL.device).float().t()
            if melody.dim() == 1:
                melody = melody[None]
            melody = melody[..., :int(sr * duration)]
            melody = convert_audio(melody, sr, target_sr, target_ac)
            processed_melodies.append(melody)

    if any(m is not None for m in processed_melodies):
        outputs = MODEL.generate_with_chroma(
            descriptions=texts,
            melody_wavs=processed_melodies,
            melody_sample_rate=target_sr,
            progress=progress,
            return_tokens=USE_DIFFUSION
        )
    else:
        outputs = MODEL.generate(texts, progress=progress, return_tokens=USE_DIFFUSION)
    if USE_DIFFUSION:
        outputs_diffusion = MBD.tokens_to_wav(outputs[1])
        outputs = torch.cat([outputs[0], outputs_diffusion], dim=0)
    outputs = outputs.detach().cpu().float()
    pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    out_videos = [pending_video.result() for pending_video in pending_videos]
    for video in out_videos:
        file_cleaner.add(video)
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs


def predict_batched(texts, melodies):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/musicgen-melody')
    res = _do_predictions(texts, melodies, BATCHED_DURATION)
    return res


def predict_full(model, decoder, text, melody, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    global USE_DIFFUSION
    INTERRUPTING = False
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    if decoder == "MultiBand_Diffusion":
        USE_DIFFUSION = True
        load_diffusion()
    else:
        USE_DIFFUSION = False
    load_model(model)

    def _progress(generated, to_generate):
        progress((min(generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, wavs = _do_predictions(
        [text], [melody], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef)
    if USE_DIFFUSION:
        return videos[0], wavs[0], videos[1], wavs[1]
    return videos[0], wavs[0], None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

openai.organization = "org-esyDe83fzs5JWkeZOuyervFy"
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def sentimentAnalysis(text):

    sentiments = {}

    softmax = Softmax(dim=1)

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model(**inputs)

    probs = softmax(outputs.logits)

    prob_class_pairs = list(zip(probs[0].tolist(), range(len(probs[0]))))

    sorted_probs = sorted(prob_class_pairs, key=lambda x: x[0], reverse=True)

    for prob, class_idx in sorted_probs:
        sentiments[model.config.id2label[class_idx]] = f"{prob*100:.2f}%"

    print("foo sentiment")
    print(sentiments)
    return sentiments

def generateTime(input_text, time):
    if time.value != 0:
       # print("This piece will last "+str(time.value)+" second(s).")
        return time.value
    else:
        timeSystem = "Your job is to estimate how long a certain text would last when spoken aloud by the average human being. Take intonation, phrasing, emphasis, and punctuation into account. When you are given a text, output the time it would take in seconds. Please do not provide any additional details, only the number value of the seconds it would take. The value must be an integer; do not include the word 'seconds' alongside it."
        timeResponse = openai.ChatCompletion.create(
        model="gpt-4",
        temperature = 1,
        messages=[
                {"role": "system", "content": timeSystem},
                {"role": "user", "content": input_text.value}
            ]
        )
        timeOutput = timeResponse.choices[0].message['content']
        #print("")
        #print("This piece will last "+timeOutput+" second(s). This is how long it would take the average person to say this text aloud.")
        return timeOutput
        
def generateParameters(sentiments, input_extra):

    parameterSystem = "You are an expert sentiment analyst and musicologist. Your expertise is in converting emotions into music. These emotions are generated from a sentiment analysis AI that converts a piece of text into a list of sentiments and a percentage distribution for how strong that sentiment is in the text. You excel at taking this information and converting it into musical parameters that correctly and accurately describe the sentiment. As a musicologist, you have a vast knowledge base and a deep understanding of the Romantic era of Western classical music, specifically the work of Ludwig Van Beethoven. The musical parameters that you generate to describe the given sentiment are accurately in line with the conventions of Romantic era composers in style, structure, and meaning. You will be given a sentiment analysis and (possibly) additional music parameters. You will convert that into musical parameters. Here's an example input: <sentiments> neutral: 64.01%, confusion: 0.01%, curiosity: 0.2%, disapproval: 0.01%, approval: 0.01%, realization: 15.67%, annoyance: 0.02%, optimism: 1.02%, disappointment: 1.89%, surprise: 0.64%, anger: 2.21%, disgust: 0.02%, love: 2.86%, caring: 0.06%, amusement: 0.01%, fear: 0%, sadness: 0%, gratitude: 0.01%, desire: 5.3%, excitement: 0.01%, joy: 0.01%, admiration: 0.01%, embarrassment: 0.1%, nervousness: 0.01%, remorse: 4.06%, grief: 1.81%, relief: 0.03%, pride: 0.01% </sentiments> A corresponding output of parameters for this example could be: <parameters> rhythm: steady; time signature: 2/2; dynamics: pianissimo; expression: legato; texture: homophonic, chords and ostinato and melody; harmony: minor; form: ostinato, A-B-A-C-A structure; tempo: Adagio sostenuto; melody: simple and elegant repeating motif; character: calm and introspective </parameters> One caveat to keep in mind is that you may also be given some additional input parameters. If these are given, they must override any parameters you come up with. For example, if this input was given alongside the sentiment analysis: <input-parameters> Allegro forte </input-parameters> The parameter output given above would have to be altered like this: <parameters> rhythm: steady; time signature: 2/2; dynamics: forte; expression: legato; texture: homophonic, chords and ostinato and melody; harmony: minor; form: ostinato, A-B-A-C-A structure; tempo: Allegro; melody: simple and elegant repeating motif; character: calm and introspective </parameters> For this process, you should think through each step by step before generating the output for the sentiments. First, organize the input into sentiments and parameters (if given). First analyze the sentiments and come up with a summary. Then, take that summary and describe in detail how those emotions could be captured by music. Remember, this description is trying to figure out how music can evoke those same emotions in humans when it's played back to them. Be creative with it! When you've determined the appropriate music parameters, I want you to write a prompt for a music-generating AI called MusicGen asking it to generate music incorporating the parameters you just came up with. You must also specify the following things: <specifications> 1. The music is for solo piano. 2. The music is in the style of Ludwig Van Beethoven. 3. The music should have a clear melodic idea with a start and end. 4. Do not specify how long (in seconds) the piece should last. </specifications> It's important to mention that this AI is not very intelligent and does not comprehend English as well as you do. You must be incredibly direct and straightforward, and the prompt should only be a paragraph. Do not refer to the AI, simply provide it accurate and concrete instructions without any flourish. When you've written this prompt, output it. That should be your only output, not the parameters or your thinking process or anything else. Good luck! "
    parameterResponse = openai.ChatCompletion.create(
        model="gpt-4-32k",
        temperature = 1,
        messages=[
            {
            "role": "system",
            "content": parameterSystem
            },
            {
                "role": "user",
                "content": str(sentiments)+" and here are the additional parameters that should override (none given if blank): "+str(input_extra.value)
            }
        ]
    )
    parameterOutput = parameterResponse['choices'][0]['message']['content']
    print("foo parameter")
    print(parameterOutput)
    return(parameterOutput)

def generatePrompt(input_text, input_extra):
    
    sent = sentimentAnalysis(str(input_text.value))
    print("foo sent")
    print(sent)
    prompt = generateParameters(sent, input_extra)

    return prompt










def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # Immortal Beloved
            This is your private program for Immortal Beloved, the software used to produce AI-generated music to be used in productions of "Man and Muse" by Rushil Byatnal.
            Models used: Audiocraft by Meta, GPT-4 by OpenAI, and RoBERTa by Meta ([trained by Sam Lowe](https://huggingface.co/SamLowe/roberta-base-go_emotions))
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Input Text", interactive=True, value = " ")
                    print("foo text")
                    print(text.value)
                    with gr.Column():
                        extra = gr.Text(label="Any additional parameters?", interactive=True, value =" ")
                        duration = gr.Slider(minimum=0, maximum=120, value=0, label="Custom Time? Set to 0 if you instead want the time automatically set based on the length of the text.", interactive=True)
                        melody = gr.Audio(source="upload", type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Submit")
                    # Adapted from https://github.com/rkfg/audiocraft/blob/long/app.py, MIT license.
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    model = gr.Radio(["facebook/musicgen-melody", "facebook/musicgen-medium", "facebook/musicgen-small",
                                      "facebook/musicgen-large"],
                                     label="Model", value="facebook/musicgen-melody", interactive=True)
                with gr.Row():
                    decoder = gr.Radio(["MultiBand_Diffusion", "Default"],
                                       label="Decoder", value="MultiBand_Diffusion", interactive=True)
                with gr.Row():
                    topk = gr.Number(label="Top-k", value=250, interactive=True)
                    topp = gr.Number(label="Top-p", value=0, interactive=True)
                    temperature = gr.Number(label="Temperature (Adjust with caution; will provide varying results)", value=.85, interactive=True)
                    cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
                diffusion_output = gr.Video(label="MultiBand Diffusion Decoder")
                audio_diffusion = gr.Audio(label="MultiBand Diffusion Decoder (wav)", type='filepath')
        submit.click(toggle_diffusion, decoder, [diffusion_output, audio_diffusion], queue=False,
                     show_progress=False).then(predict_full, inputs=[model, decoder, gr.Text(value=str(generatePrompt(text, extra))), melody, gr.Slider(value=generateTime(text, duration)), topk, topp,
                                                                     temperature, cfg_coef],
                                               outputs=[output, audio_output, diffusion_output, audio_diffusion])

        gr.Markdown(
            """
            ### More details

            The model will generate a short musical extract based on the text provided. This is done through a sentiment
            analysis of the text and an attempt to recreate the same sentiment profile through music parameters. In other words,
            this is the computer's attempt to translate your text into music (currently restricted to solo piano in the 
            style of Beethoven).



            The following details are provided by Meta regarding their Audiocraft model:

            The model can generate up to 30 seconds of audio in one pass. It is now possible
            to extend the generation by feeding back the end of the previous chunk of audio.
            This can take a long time, and the model might lose consistency. The model might also
            decide at arbitrary positions that the song ends.

            **WARNING:** Choosing long durations will take a long time to generate (2min might take ~10min).
            An overlap of 12 seconds is kept with the previously generated chunk, and 18 "new" seconds
            are generated each time.

            We present 4 model variations:
            1. facebook/musicgen-melody -- a music generation model capable of generating music condition
                on text and melody inputs. **Note**, you can also use text only.
            2. facebook/musicgen-small -- a 300M transformer decoder conditioned on text only.
            3. facebook/musicgen-medium -- a 1.5B transformer decoder conditioned on text only.
            4. facebook/musicgen-large -- a 3.3B transformer decoder conditioned on text only.

            We also present two way of decoding the audio tokens
            1. Use the default GAN based compression model
            2. Use MultiBand Diffusion from (paper linknano )

            When using `facebook/musicgen-melody`, you can optionally provide a reference audio from
            which a broad melody will be extracted. The model will then try to follow both
            the description and melody provided.

            You can also use your own GPU or a Google Colab by following the instructions on our repo.
            See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
            for more details.
            """
        )

        interface.queue().launch(**launch_kwargs)


def ui_batched(launch_kwargs):
    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # MusicGen

            This is the demo for [MusicGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for music generation
            presented at: ["Simple and Controllable Music Generation"](https://huggingface.co/papers/2306.05284).
            <br/>
            <a href="https://huggingface.co/spaces/facebook/MusicGen?duplicate=true"
                style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank">
            <img style="margin-bottom: 0em;display: inline;margin-top: -.25em;"
                src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
            for longer sequences, more control and no queue.</p>
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    text = gr.Text(label="Describe your music", lines=2, interactive=True, value="")
                    print("foo text")
                    print(text)
                    with gr.Column():
                        radio = gr.Radio(["file", "mic"], value="file",
                                         label="Condition on a melody (optional) File or Mic")
                        melody = gr.Audio(source="upload", type="numpy", label="File",
                                          interactive=True, elem_id="melody-input")
                with gr.Row():
                    submit = gr.Button("Generate")
            with gr.Column():
                output = gr.Video(label="Generated Music")
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')
        submit.click(predict_batched, inputs=[text, melody],
                     outputs=[output, audio_output], batch=True, max_batch_size=MAX_BATCH_SIZE)
        radio.change(toggle_audio_src, radio, [melody], queue=False, show_progress=False)
        gr.Examples(
            fn=predict_batched,
            examples=[
                [
                    "An 80s driving pop song with heavy drums and synth pads in the background",
                    "./assets/bach.mp3",
                ],
                [
                    "A cheerful country song with acoustic guitars",
                    "./assets/bolero_ravel.mp3",
                ],
                [
                    "90s rock song with electric guitar and heavy drums",
                    None,
                ],
                [
                    "a light and cheerly EDM track, with syncopated drums, aery pads, and strong emotions bpm: 130",
                    "./assets/bach.mp3",
                ],
                [
                    "lofi slow bpm electro chill with organic samples",
                    None,
                ],
            ],
            inputs=[text, melody],
            outputs=[output]
        )
        gr.Markdown("""
        ### More details

        The model will generate 12 seconds of audio based on the description you provided.
        You can optionally provide a reference audio from which a broad melody will be extracted.
        The model will then try to follow both the description and melody provided.
        All samples are generated with the `melody` model.

        You can also use your own GPU or a Google Colab by following the instructions on our repo.

        See [github.com/facebookresearch/audiocraft](https://github.com/facebookresearch/audiocraft)
        for more details.
        """)

        demo.queue(max_size=8 * 4).launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    # Show the interface
    if IS_BATCHED:
        global USE_DIFFUSION
        USE_DIFFUSION = False
        ui_batched(launch_kwargs)
    else:
        ui_full(launch_kwargs)
