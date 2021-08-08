"""
Dash Board for image on CPU
"""

import streamlit as st
from PIL import Image
import time
import os
st.set_page_config(layout='wide', initial_sidebar_state='collapsed')


def clean_dir(dirname):
	"""
	Clean the file.
	Only cleans the files, not the folders inside.
	"""
	lit = os.listdir(dirname)
	for i in lit:
		try:
			os.remove(dirname + '\\' + i)
		except Exception as e:
			print(e)


st.title("Self Supervised Attention")

st.info("DINO - Self **Di**stillation with **No** Labels")
acceleration = st.radio("Select the type of Accelerator: ", ["cuda", "cpu"])
st.write("Initializing acceleration on {}".format(acceleration))

uploaded_file = st.file_uploader("Choose an image (should be in jpg) format", type="jpg")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Image.', width=500)
	st.write("")
	image.save("Logs/SavedImg.jpg")

	execution = st.button("Start the Process")

	if execution:
		with st.spinner("Processing on {} | ETA: Calculating...".format(acceleration)):
			start = time.time()
			clean_dir("output")
			os.system("python visualize_attention.py \
			--pretrained_weights data/dino_deitsmall8_pretrain_full_checkpoint.pth \
			--image_path Logs/SavedImg.jpg --output_dir output")

			st.write("success")

		duration = time.time() - start
		st.info("Generation Time: {} secs".format(duration))



		image0 = Image.open(r"output/img.png")
		image1 = Image.open(r"output/attn-head0.png")
		# st.image(image1, caption="Attention layers part 1")

		image2 = Image.open(r"output/attn-head1.png")
		# st.image(image2, caption="Attention layers part 2")

		image3 = Image.open(r"output/attn-head2.png")
		# st.image(image, caption="Attention layers part 3")

		image4 = Image.open(r"output/attn-head3.png")
		# st.image(image, caption="Attention layers part 4")

		image5 = Image.open(r"output/attn-head4.png")
		# st.image(image, caption="Attention layers part 5")
		captions = ["**iteration-0**", "**iteration-1**", "**iteration-2**", "**iteration-3**", "**iteration-4**", "**iteration-5 (Final)**"]
		st.image([image0, image5, image4, image3, image2, image1], width=300, caption=captions)

		if st.button("Reset Head"):
			clean_dir("output")
			st.info("Image weights and dir cleaned!")


hide_stream_lit_style = """
			<style>
			#MainMenu {visibility: hidden;}
			footer {visibility: hidden;}
			footer:after {
				content:'Ashutosh'; 
				visibility: visible;
				display: block;
				position: relative;
				# background-color: green;
				# foreground-color: blue;
				padding: 5px;
				top: 5px;
			}
			</style>
			"""

st.markdown(hide_stream_lit_style, unsafe_allow_html=True)
