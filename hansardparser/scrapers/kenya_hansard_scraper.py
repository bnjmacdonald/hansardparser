"""Scrapes new Hansard transcripts from the Kenya Parliament website.

Example usage::

    scraper = HansardScraper(
        outpath=settings.MEDIA_ROOT,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36'},
        wait=5,
        max_retries=3,
        verbose=0
    )
    scraper.get_hansards(get_all=True)

Command line usage::

    python -m hansardparser.scrapers.kenya_hansard_scraper -v 1 -o data/raw/hansards/2013-

Contributors:

- Zacharia Mwangi contributed to this script.

Todos:

    TODO: save url link to downloaded transcript so that it can be linked to
        for users who want to see the original. (could save this in a csv with
        the transcript name and associated url).
    
    TODO: add command line option for user to specify whether to overwrite
        existing PDFs or skip downloading PDFs if the PDF already exists.
"""

import os
import time
import re
import argparse
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter

import settings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=0,
        help="verbosity (integer)"
    )
    parser.add_argument(
        '-n',
        '--num_pages',
        type=int,
        help="Only retrieve the last n pages of transcripts."
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        dest='output',
        default=os.path.join(settings.DATA_ROOT, 'raw', 'transcripts'),
        help='Output directory for scraped Hansards',
    )
    args = parser.parse_args()
    return args

def scrape():
    args = parse_args()
    scraper = HansardScraper(
        outpath=args.output,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.101 Safari/537.36'},
        wait=3,
        max_retries=3,
        verbose=args.verbosity
    )
    scraper.get_hansards(num_pages=args.num_pages)

class HansardScraper(object):
    """Defines the HansardScraper class, which contains methods for scraping 
    Hansards from the Parliament of Kenya website.
    """

    domain = 'http://www.parliament.go.ke'

    def __init__(self, outpath, headers=None, proxies=None, timeout=30, wait=5, max_retries=0, verbose=0):
        self.outpath = outpath
        self.session = requests.Session()
        self.session.mount(self.domain, HTTPAdapter(max_retries=max_retries))
        if proxies is not None:
            self.session.proxies = proxies
        if headers is not None:
            self.session.headers = headers
        self.verbose = verbose
        self.timeout = timeout
        # self.max_retries = max_retries
        self.wait = wait

    def get_hansards(self, num_pages=None):
        """downloads Hansard PDFs.

        Arguments:

            num_pages: int (default: None). Maximum number of pages to scrape
                data from. If None, scrapes transcripts from all pages.
        """
        orig_files = os.listdir(self.outpath)
        relative_url = 'the-national-assembly/house-business/hansard'
        r = self._get_request(relative_url, params={})
        soup = BeautifulSoup(r.text, 'html5lib')
        items = self._get_items(soup)
        self._process_items(items)
        next_page_url = self._get_next_page_link(soup)
        page = 1
        while next_page_url and (num_pages is None or page < num_pages):
            r = self._get_request(next_page_url, params={})
            soup = BeautifulSoup(r.text, 'html5lib')
            items = self._get_items(soup)
            self._process_items(items)
            next_page_url = self._get_next_page_link(soup)
            page += 1
        if self.verbose:
            new_files = list(set(os.listdir(self.outpath)) - set(orig_files))
            print('Downloaded {0} new files. Total files: {1}. Files downloaded: {2}'.format(len(new_files), len(os.listdir(self.outpath)), '\n'.join(new_files)))
        return 0

    def _get_items(self, soup):
        """gets list of items (e.g. list of <li> tags with Hansard PDF links).

        Returns list of bs4.Tag objects.
        """
        container = soup.find(id='k2Container')  # note: probably unnecessary, but is a useful measure for making sure that the correct list of items is found.
        items = container.find(id='itemListPrimary').find_all('div', {'class': 'itemContainer'})
        if self.verbose > 1:
            print('Found {0} items'.format(len(items)))
        return items

    def _get_request(self, relative_url='', params={}):
        """returns a HTTP response based on a url and set of GET parameters."""
        if relative_url.startswith('/'):
            relative_url = relative_url[1:]
        if self.verbose:
            print('Making GET request to: {0}'.format(os.path.join(self.domain, relative_url)))
        r = self.session.get(os.path.join(self.domain, relative_url), params=params, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError('{0} returned with status code {1}'.format(r.url, r.status_code))
        time.sleep(self.wait)
        return r

    def _process_items(self, items):
        """wrapper to self._process_items."""
        if not len(items):
            raise RuntimeError('There are no items to process.')
        for item in items:
            self._process_item(item)
        return 0

    def _process_item(self, item):
        """visits href in item and downloads file (if doesn't exist already).
        """
        links = item.find_all('a')
        if len(links) > 1:
            print('WARNING: more than one <a> tag found in list item.')
        r = self._get_request(links[0].attrs['href'])
        fname = self._get_filename(r.headers['Content-Disposition'])
        self._download_file(r, fname)
        return 0

    def _get_filename(self, s):
        """extracts filename from Content-Disposition string (s)."""
        regex_filename = re.search(r'filename=(?P<filename>.+\.pdf)', s)
        fname = regex_filename.group('filename')
        return fname

    def _download_file(self, r, fname):
        """downloads a file from a request if it doesn't exist already."""
        # skip file if it already exists.
        if os.path.isfile(os.path.join(self.outpath, fname)):
            if self.verbose:
                print('Skipping over file: {0} (already exists)'.format(fname))
            return None
        with open(os.path.join(self.outpath, fname), 'wb') as f:
            f.write(r.content)
        if self.verbose:
            print('Saved {0} to {1}.'.format(fname, self.outpath))
        return 0

    def _get_next_page_link(self, soup):
        """gets the relative href for the next page."""
        next_page_link = soup.find('div', {'class': 'k2Pagination'}).find('li', {'class': 'pagination-next'}).a
        if next_page_link is not None:
            return next_page_link.attrs['href']
        return None

if __name__ == '__main__':
    scrape()
