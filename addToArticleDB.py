'''

Site Scraper:
    init:   (source_name, start_url, right_score, allowed_domain=None, base_url=None, cycle_limit=3)
    scrape: (json_filename=None, tsv_filename=None, db_filename=None)


source list format :
    {
        'source_name':'',
        'start_url':'',
        'right_score':'',
        'allowed_domain':'',
        'base_url':'',
    }

'''
import random
import multiprocessing as mp

from SiteScraper import SiteScraper

db_path = 'articles.db'

left_sources = [
    {
        'source_name':'msnbc',
        'start_url':'https://www.msnbc.com/',
        'right_score':'0',
        'allowed_domain':'msnbc.com',
        'base_url':'https://www.msnbc.com'
    },
    {
        'source_name':'vox',
        'start_url':'https://www.vox.com/',
        'right_score':'0',
        'allowed_domain':'vox.com',
        'base_url':'https://www.vox.com'
    },
    {
        'source_name':'jacobin',
        'start_url':'https://jacobinmag.com/',
        'right_score':'0',
        'allowed_domain':'jacobinmag.com',
        'base_url':'https://jacobinmag.com'
    },
    {
        'source_name':'huffpo',
        'start_url':'https://www.huffpost.com/news/politics',
        'right_score':'0',
        'allowed_domain':'huffpost.com',
        'base_url':'https://www.huffpost.com'
    },
    {
        'source_name':'buzzfeed',
        'start_url':'https://www.buzzfeednews.com/',
        'right_score':'0',
        'allowed_domain':'buzzfeednews.com',
        'base_url':'https://www.buzzfeednews.com'
    },
    {
        'source_name':'newyorker',
        'start_url':'https://www.newyorker.com/',
        'right_score':'0',
        'allowed_domain':'newyorker.com',
        'base_url':'https://www.newyorker.com'
    },
    {
        'source_name':'motherjones',
        'start_url':'https://www.motherjones.com/',
        'right_score':'0',
        'allowed_domain':'motherjones.com',
        'base_url':'https://www.motherjones.com'
    },
    {
        'source_name':'washingtonpost',
        'start_url':'https://www.washingtonpost.com/',
        'right_score':'0',
        'allowed_domain':'washingtonpost.com',
        'base_url':'https://www.washingtonpost.com'
    },
    {
        'source_name':'democracynow',
        'start_url':'https://www.democracynow.org/',
        'right_score':'0',
        'allowed_domain':'democracynow.org',
        'base_url':'https://www.democracynow.org'
    },
    {
        'source_name':'salon',
        'start_url':'https://www.salon.com/category/politics',
        'right_score':'0',
        'allowed_domain':'salon.com',
        'base_url':'https://www.salon.com/'
    }
]



right_sources = [
    {
        'source_name':'breitbart',
        'start_url':'https://www.breitbart.com/',
        'right_score':'1',
        'allowed_domain':'breitbart.com',
        'base_url':'https://www.breitbart.com'
    },
    {
        'source_name':'fox',
        'start_url':'https://www.foxnews.com/',
        'right_score':'1',
        'allowed_domain':'foxnews.com',
        'base_url':'https://www.foxnews.com'
    },
    {
        'source_name':'theblaze',
        'start_url':'https://www.theblaze.com/',
        'right_score':'1',
        'allowed_domain':'theblaze.com',
        'base_url':'https://www.theblaze.com'
    },
    {
        'source_name':'hanity',
        'start_url':'https://hannity.com/',
        'right_score':'1',
        'allowed_domain':'hannity.com',
        'base_url':'https://hannity.com/'
    },
    {
        'source_name':'drudgereport',
        'start_url':'https://www.drudgereport.com/',
        'right_score':'1',
        'allowed_domain':'drudgereport.com',
        'base_url':'https://www.drudgereport.com'
    },
    {
        'source_name':'natrev',
        'start_url':'https://www.nationalreview.com/',
        'right_score':'1',
        'allowed_domain':'nationalreview.com',
        'base_url':'https://www.nationalreview.com'
    },
    {
        'source_name':'redstate',
        'start_url':'https://www.redstate.com/',
        'right_score':'1',
        'allowed_domain':'redstate.com',
        'base_url':'https://www.redstate.com'
    },
    {
        'source_name':'federalist',
        'start_url':'https://thefederalist.com/',
        'right_score':'1',
        'allowed_domain':'thefederalist.com',
        'base_url':'https://thefederalist.com'
    },
    {
        'source_name':'twitchy',
        'start_url':'https://twitchy.com/',
        'right_score':'1',
        'allowed_domain':'twitchy.com',
        'base_url':'https://twitchy.com'
    },
    {
        'source_name':'weeklystandard',
        'start_url':'https://www.weeklystandard.com/',
        'right_score':'1',
        'allowed_domain':'weeklystandard.com',
        'base_url':'https://www.weeklystandard.com'
    }
]


def scrape_wrapper(kwargs, output):
    try:
        scraper = SiteScraper(**kwargs)
        scraper.scrape(db_filename=db_path)
    except Exception as e:
        print('Error scraping source: %s' % kwargs['source_name'])
        print(e)
    else:
        articles_added = len(scraper.data)
        print('Adding %5i article(s) from %15s' % (articles_added, kwargs['source_name']))
        output.put(articles_added)
    return



def run_scrape():
    p_size = 3

    random.shuffle(left_sources)
    random.shuffle(right_sources)
    left_articles_added = 0

    for i in range(0, len(left_sources), p_size):
        s1, s2, s3 = left_sources[i:i+3]
        output = mp.Queue()
        processes = [
            mp.Process(target=scrape_wrapper, args=(s1, output)),
            mp.Process(target=scrape_wrapper, args=(s2, output)),
            mp.Process(target=scrape_wrapper, args=(s3, output))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        left_articles_added += sum([output.get() for p in processes])

    for i in range(0, len(right_sources), p_size):
        s1, s2, s3 = right_sources[i:i+3]
        output = mp.Queue()
        processes = [
            mp.Process(target=scrape_wrapper, args=(s1, output)),
            mp.Process(target=scrape_wrapper, args=(s2, output)),
            mp.Process(target=scrape_wrapper, args=(s3, output))
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        right_articles_added += sum([output.get() for p in processes])
        
    print('\n\n')
    print('FINAL RESULTS: Added a total of %7i left articles and %7i right articles.' % (left_articles_added, right_articles_added))
    return


if __name__ == '__main__':
    run_scrape()