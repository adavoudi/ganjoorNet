import scrapy

class GanjoorSpider(scrapy.Spider):

        name = "ganjoor"

        def start_requests(self):
            base_url = 'http://ganjoor.net/ferdousi/shahname/'
            yield scrapy.Request(url=base_url, callback=self.parse)

        def parse(self, response):
            mesras = response.css('.b p::text').extract()
            if len(mesras) == 0: # then this is not a sher page
                for url in response.css('p a::attr(href)').extract():
                    yield scrapy.Request(url, callback=self.parse)
            else:
                filename = 'shahname/' + response.url[response.url.index('ferdousi/'):-1].replace('/','_') + ".txt"
                file = open(filename, 'w');
                for index in range(0,len(mesras),2):
                    file.write(mesras[index].encode('utf8'))
                    file.write('\t'.encode('utf8'))
                    file.write(mesras[index+1].encode('utf8'))
                    file.write('\n'.encode('utf8'))
                file.close()
