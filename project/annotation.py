"""annotation.py: gathers helper functions for annotation"""
import io
import os
import shutil
from typing import List
import cv2 as cv

import ipywidgets as widgets
import numpy as np
import pandas as pd
from ipywidgets import Layout, GridspecLayout, HBox, VBox
from PIL import Image


class AnnotationUI:
    """
    Class for storing the UI with callbacks for annotation.

    Args:
        coins (List[np.ndarray]): list coins as arrays
    """

    euro_options = ['2 EUR', '1 EUR', '0.5 EUR', '0.2 EUR', '0.1 EUR', '0.05 EUR', '0.02 EUR', '0.01 EUR']
    franc_options = ['5 CHF', '2 CHF', '1 CHF', '0.5 CHF', '0.2 CHF', '0.1 CHF', '0.05 CHF']

    def __init__(self, coins: List[np.ndarray]):

        self.i = 0
        self.coins = coins

        # create an image widget from the byte data
        self.image_widget = widgets.Image(
            value=self._get_image_bytes(),
            format='png',
            layout=Layout(height='auto', width='300px')
        )

        self.grid = GridspecLayout(4, 3, height='300px')

        # initiate classification and CTA button
        self._initiate_buttons()
        self._initiate_cta()

        self.labels = []

        self.output = widgets.Output()  # for printing final text

    def display(self):
        """
        Return the UI, which consists of:
            - image of the coin;
            - buttons for classification;
            - CTA button for saving the annotation.

        Returns:
            grid (GridSpecLayout): essentially the UI
        """
        self.grid[:3, 1:] = self.buttons_vertical
        self.grid[:, 0] = self.image_widget
        self.grid[3, 2] = self.cta_row

        return widgets.VBox([self.grid, self.output])

    def on_cta_click(self, button):
        """Callback for saving the annotation and updating the UI."""
        # get annotation value and add it to annotations
        toggle_values = [self.euro_buttons.value, self.franc_buttons.value, self.ood_buttons.value]
        annotation = [v for v in toggle_values if v is not None][0]  # only 1 non-None could exist
        self.labels.append(annotation)

        # reset toggle buttons
        self.euro_buttons.value = None
        self.franc_buttons.value = None
        self.ood_buttons.value = None

        self.i += 1

        if self.i == len(self.coins):  # if all pictures have been annotated

            self.grid.height = '0px'  # hide UI

            # create directory for annotations
            current_dir = os.getcwd()
            base_dir = os.path.join(current_dir, r'train_annotated')

            if os.path.exists(base_dir):
                shutil.rmtree(base_dir)  # remove directory if exists

            os.makedirs(base_dir)

            # create DataFrame for labels
            labels_df = pd.DataFrame(columns=['filename', 'label'])

            for i in range(len(self.coins)):
                filename = f'coin_{i}.png'
                cv.imwrite(os.path.join(base_dir, filename), self.coins[i])

                new_row = {'filename': filename, 'label': self.labels[i].strip()}
                labels_df.loc[len(labels_df)] = new_row

            labels_df.to_csv('train_annotated/labels.csv')

            with self.output:
                print("No images left to annotate. "
                      "Images with their corresponding labels were saved to 'train_annotated/'.")

        else:
            self.image_widget.value = self._get_image_bytes()  # just get new picture

    def on_btn_change(self, change):
        """Callback for keeping only one class selected among different ToggleButtons."""
        if change['name'] == 'value' and change['new']:
            # check which button group was changed
            if change['owner'] == self.euro_buttons:
                self.franc_buttons.value, self.ood_buttons.value = None, None
            elif change['owner'] == self.franc_buttons:
                self.euro_buttons.value, self.ood_buttons.value = None, None
            elif change['owner'] == self.ood_buttons:
                self.euro_buttons.value, self.franc_buttons.value = None, None

    def _initiate_cta(self):
        """Initiate call-to-action button and wrap it with HBox."""
        button_layout = Layout(height='50px', width='200px')
        self.cta = widgets.Button(description='Annotate', button_style='info', layout=button_layout)
        self.cta.on_click(self.on_cta_click)

        # HBox layout configuration for aligning the button at the bottom-right
        box_layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            justify_content='flex-end',
            align_items='flex-end',
            height='auto',
            width='auto'
        )

        # container with the layout applied, containing the button
        self.cta_row = HBox(children=[self.cta], layout=box_layout)

    def _get_image_bytes(self):
        """
        Since widgets.Image requires certain byte format, take the
            current image and return the byte-version.
        """
        img = Image.fromarray(self.coins[self.i][:, :, ::-1])  # bgr -> rgb

        # convert PIL Image to a displayable format, e.g., PNG
        with io.BytesIO() as f:
            img.save(f, format='PNG')
            image_bytes = f.getvalue()

        return image_bytes

    def _initiate_buttons(self):
        """Initiate toggle buttons for annotating data."""

        self.euro_buttons = widgets.ToggleButtons(
            options=self.euro_options,
            description='Euro:'
        )

        self.franc_buttons = widgets.ToggleButtons(
            options=self.franc_options,
            description='Franc:'
        )

        self.ood_buttons = widgets.ToggleButtons(
            options=['OOD'],
            description='Ood:'
        )

        for btn in [self.euro_buttons, self.franc_buttons, self.ood_buttons]:
            btn.style.button_width = '80px'
            btn.index = None

        self.buttons_vertical = VBox(
            [self.euro_buttons, self.franc_buttons, self.ood_buttons],
            layout=widgets.Layout(justify_content='center', height='auto')
        )

        # add callbacks to only keep one btn selected
        self.euro_buttons.observe(self.on_btn_change, names='value')
        self.franc_buttons.observe(self.on_btn_change, names='value')
        self.ood_buttons.observe(self.on_btn_change, names='value')