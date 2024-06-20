"""This module contains the script to process the vlm json file with the raw answers of LLaVA.

"""


import json
import re


def process_vlm_answers(input_path: str,
                        out_path: str = None):
    """This function processes the raw LLaVA answers to become a List containing the Lists with the seperated answers.

    Args:
        input_path: The file path of the json file containing a list of raw LLaVA answers.
        out_path: The path of the json file that the output is saved to. If None the output will be returned. Defaults to None.

    Returns:
        None if an out_path is specified otherwise it outputs the list of lists containing the seperate answers, 
        together with the image filepath the answers belong to.
    """    
    # loading the json:
    with open(input_path, 'r') as f:
        rawtext = json.load(f)

    # Removing all of the \n's from the answers
    cleanrawtext = [txt.replace('\n', '') for txt in rawtext ]

    # Splitting the answers to be seperated
    answerssplit = [re.split('Answer [1-5]: ', txt) for txt in cleanrawtext]

    # Check if all answers where in the correct format (If not change by hand!)
    for i,sample in enumerate(answerssplit):
        if sample.__len__() != 6:
            print(i, sample.__len__(), sample)


    # Replacing N/A answers with a natural language equavalent:
    for sample in answerssplit:
        sample[0] = sample[0][:-1]  #remove the | afther the path
        if sample[2] == 'N/A':
            sample[2] = 'There is no interaction that a person is having with an object.'
        if sample[3] == 'N/A':
            sample[3] = 'There is no object that a person is having an interacting with.'
        if sample[4] == 'N/A':
            sample[4] = 'There is no interaction that a person is having with an object.'
        if sample[5] == 'N/A':
            sample[5] = 'There is no interaction that a person is having with an object.'

    if out_path == None:
        return answerssplit
    
    with open(out_path, 'w') as f:
        json.dump(answerssplit, f)


if __name__ == '__main__':
    # Train:
    process_vlm_answers('VLM_Answers/Raw_Answers/vlm_answers_train.json', 'VLM_Answers/Processed_Answers/vlm_answers_train_processed.json')
    # Test:
    process_vlm_answers('VLM_Answers/Raw_Answers/vlm_answers_test.json', 'VLM_Answers/Processed_Answers/vlm_answers_test_processed.json')