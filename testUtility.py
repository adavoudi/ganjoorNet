import my_txtutils as txt

codetext, valitext, bookranges = txt.read_data_files('ganjoor-scrapy/shahname/', validation=True)

print bookranges[0]
print txt.decode_to_text(codetext[bookranges[0]['start']:bookranges[0]['end']])
