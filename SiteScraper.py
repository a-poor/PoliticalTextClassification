"""

"""

import re
from time import localtime, strftime, sleep
import json
import sqlite3
import sys

import requests
from bs4 import BeautifulSoup

class SiteScraper:
    def __init__(self, source_name, start_url, right_score, allowed_domain=None, base_url=None, cycle_limit=3, verbose=False):
        self.source_name = source_name
        self.start_url = start_url
        self.verbose = verbose
        right_score = int(right_score)
        assert right_score in [0,1], 'Right score should be either 0 or 1'
        self.right_score = right_score
        self.left_score = 1 - right_score
        if allowed_domain:
            self.allowed_domain = allowed_domain
        else:
            partial_url = re.findall(r'\..*\.\w{3}', allowed_domain)[0]
            if partial_url.startswith('.'):
                self.allowed_domain = partial_url[1:]
            else:
                self.allowed_domain = partial_url
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = 'http://www.' + self.allowed_domain
        
        self.cycle_limit = cycle_limit

        # Create some attributes to store the urls    
        self.links_searched = set()
        self.links_searching = set()
        self.links_to_search = set()
        self.links_to_search.add(start_url)

        # Create an attribute to store the data pulled
        self.data = []
        return

    def timestamp(self):
        return strftime('%y/%m/%d %H:%M:%S', localtime())

    def clean_text(self, unclean_text):
        # Turn all whitespace into single spaces
        text = re.sub(r'\s+', ' ', unclean_text)
        # Remove all weird characters
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def add_data(self, url, text):
        self.data.append({
            'timestamp': self.timestamp(),
            'source': self.source_name,
            'url': url,
            'text': self.clean_text(text)
        })
        return

    def soup_page(self, url):
        response = requests.get(url)
        return BeautifulSoup(response.content, 'lxml')

    def parse_Ps(self, soup):
        ps = soup.find_all('p')
        p_text = [p.text for p in ps]
        return ' '.join(p_text)

    def parse_As(self, soup):
        As = soup.find_all('a')
        try:
            all_links = [link.attrs['href'] for link in As if 'href' in link.attrs]
        except:
            if self.verbose:
                print(As[0].attrs)
            raise 
        for link in all_links:
            # clean_link = self.process_link(link)
            if link not in self.links_searched or link not in self.links_searching:
                    self.links_to_search.add(link)
                    
        return

    def process_link(self, link):
        if link in ['', '/', '//']:
            # self.links_searched.add(link)
            return None
        elif self.allowed_domain in link:
            if link.startswith('http'):
                return link
            else:
                return 'https:' + link
        elif link.startswith('/'):
            return self.base_url + link
        elif re.search(r'\.\w*%s\w*\.' % self.source_name, link):
            if link.startswith('http'):
                return link
            else:
                return 'https:' + link
        else:
            # print('Couldn\'t parse link: ', link)
            return None

    def parse_page(self, url):
        soup = self.soup_page(url)
        self.parse_As(soup)
        page_text = self.parse_Ps(soup)
        return page_text

    def single_scrape(self):
        # Make sure to clear out all queued links
        while len(self.links_searching) > 0:
            self.links_searched.add(self.links_searching.pop())
        # Move all on-deck links to the active queue
        while len(self.links_to_search) > 0:
            self.links_searching.add(self.links_to_search.pop())
        # Process links in the queue
        
        if self.verbose:
            n_links = len(self.links_searching)
            toolbar_width = 100
            sys.stdout.write('[%s]' % (' ' * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write('\b' * (toolbar_width+1))
            last_bar = -1

        for i, link in enumerate(self.links_searching):
            if self.verbose:
                current_bar = (i / n_links * 100) // 1
                if current_bar > last_bar:
                    sys.stdout.write('=')
                    sys.stdout.flush()
                    last_bar = current_bar
            clean_link = self.process_link(link)
            if clean_link is None:
                self.links_searched.add(link)
                continue
            try:
                page_text = self.parse_page(clean_link)
            except requests.HTTPError:
                if self.verbose:
                    print('<HTTPError> Sleeping')
                sleep(10)
            except Exception as e:
                if 'HTTPConnectionPool' in str(e):
                    if self.verbose:
                        print('<HTTPConnectionPool Error> Sleeping')
                    sleep(10)
                else:
                    if self.verbose:
                        print(e)
                continue
            self.add_data(link, page_text)
        # Finish by moving all queued links to the history
        while len(self.links_searching) > 0:
            self.links_searched.add(self.links_searching.pop())
        if self.verbose:
            print()
        return

    def scrape(self, json_filename=None, tsv_filename=None, db_filename=None):
        # Parse article
        i = 0
        if self.verbose:
            print('Starting scrape.')
        while i < self.cycle_limit:
            print('\tSource: %-20s | Round: %3i | Links Parsing: %5i' % (self.source_name, i, len(self.links_to_search)))
            self.single_scrape()
            if len(self.links_to_search) == 0:
                if self.verbose:
                    print('Ran out of links to parse for %10s at cycle %3i' % (self.source_name, i))
                break
            i += 1
        if self.verbose:
            print('Ending scrape. Successfully added %i pages of %i.' % (len(self.data), len(self.links_searched)))
        # Check if you need to save out the data
        if [json_filename, tsv_filename, db_filename].count(None) == 3 and __name__ == '__main__':
            print(self.data)
        else:
            if json_filename is not None:
                self.to_json(json_filename)
            if tsv_filename is not None:
                self.to_tsv(tsv_filename)
            if db_filename is not None:
                self.to_sqlite(db_filename)
        return

    def to_json(self, filename):
        assert len(self.data) > 0, "ERROR: Can't save! No data collected!"
        with open(filename, 'x') as f:
            json.dump(self.data, f)
        return

    def to_tsv(self, filename):
        assert len(self.data) > 0, "ERROR: Can't save! No data collected!"
        with open(filename, 'x') as f:
            f.write('\t'.join(['timestamp', 'source', 'url', 'text']))
            f.write('\n')
            for row in self.data:
                for col in ['timestamp', 'source', 'url', 'text']:
                    f.write(row[col])
                    if col != 'text':
                        f.write('\t')
                    else:
                        f.write('\n')
        return

    def to_sqlite(self, filename):
        assert len(self.data) > 0, "ERROR: Can't save! No data collected!"
        db = sqlite3.connect(filename)
        c = db.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS "SiteScrape"("Timestamp" TEXT, "Source" TEXT, "Url" TEXT, "Text" TEXT, "Right_Score" INTEGER, "Left_Score" INTEGER, PRIMARY KEY( "Source", "Url", "Text"));')
        db.commit()
        for row in self.data:
            if len(row['text']) == 0:
                continue
            try:
                c.execute(
                    'INSERT INTO "SiteScrape" ( "Timestamp", "Source", "Url", "Text", "Right_Score", "Left_Score" ) VALUES (?, ?, ?, ?, ?, ?);',
                    (row['timestamp'], row['source'], row['url'], row['text'], self.right_score, self.left_score)
                )
            except:
                pass
            db.commit()
        return