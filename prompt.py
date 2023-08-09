import os
import openai
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn import Softmax



time = 0

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

openai.organization = "org-esyDe83fzs5JWkeZOuyervFy"
openai.api_key = "OPENAI_API_KEY"
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

    return sentiments


# def customT(text):
#     input_time = input("Custom time in seconds? (y/n): ")

#     if input_time == "y":
#         time = input("Enter time in seconds (must be a whole number): ")
#         try:
#             timeInt = int(time)
#             if timeInt < 0:  # check for negative integers
#                 raise ValueError("The number must be a positive integer!")
#             print(f"The piece will last {time} second(s).")
#         except ValueError as e:
#             print(e)
#         return time
#     elif input_time == "n":
#         return generateTime(text)
#     else:
#         raise ValueError("Invalid input! Please enter either 'y' or 'n'.")


# def customP():
#     input_extra = input("Custom parameters? (y/n): ")
#     if input_extra == "y":
#         extra = input("Enter parameters: ")
#         #print(f"Parameters: {extra}")
#         return extra
#     elif input_extra == "n":
#         return " "
#     else:
#         raise ValueError("Invalid input! Please enter either 'y' or 'n'.")

def generateTime(input_text, time):
    if time != 0:
       # print("This piece will last "+str(time.value)+" second(s).")
        return time
    else:
        timeSystem = "Your job is to estimate how long a certain text would last when spoken aloud by the average human being. Take intonation, phrasing, emphasis, and punctuation into account. When you are given a text, output the time it would take in seconds. Please do not provide any additional details, only the number value of the seconds it would take. The value must be an integer; do not include the word 'seconds' alongside it."
        timeResponse = openai.ChatCompletion.create(
        model="gpt-4",
        temperature = 1,
        messages=[
                {"role": "system", "content": timeSystem},
                {"role": "user", "content": input_text}
            ]
        )
        timeOutput = timeResponse.choices[0].message['content']
        #print("")
        #print("This piece will last "+timeOutput+" second(s). This is how long it would take the average person to say this text aloud.")
        return timeOutput

def generatePrompt(sentiments, input_extra):

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
                "content": str(sentiments)+" and here are the additional parameters that should override (none given if blank): "+input_extra.value
            }
        ]
    )

    parameterOutput = parameterResponse['choices'][0]['message']['content']
    return(parameterOutput)

