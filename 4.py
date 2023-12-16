import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import requests
from bs4 import BeautifulSoup
import tldextract
import urllib.parse
import numpy as np


def get_links(url):
    r = requests.get(url)
    if 'text/html' not in r.headers['Content-Type']:
        return []
    html_content = r.text
    soup = BeautifulSoup(html_content, 'html.parser')
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links


def crawl(start_url, max_pages):
    visited = set()
    to_visit = [start_url]
    links = {start_url: set()}
    lock = Lock()

    def visit_url(url, n):
        nonlocal visited, to_visit, links
        print(f'Now visiting: {url} ({n}/{max_pages})')
        try:
            new_links = get_links(url)
            with lock:
                visited.add(url)
                for link in new_links:
                    parsed = urllib.parse.urlparse(link)
                    # If the link is a relative link
                    if parsed.netloc == '':
                        link = urllib.parse.urljoin(url, link)
                    # Check if the link leads to the same website
                    if tldextract.extract(url).registered_domain != tldextract.extract(link).registered_domain:
                        return
                    if link not in links:
                        links[link] = set()
                    links[url].add(link)
                    if link not in visited:
                        to_visit.append(link)
        except Exception as e:
            print(f'Error visiting {url}: {e}')

    n = 0
    with ThreadPoolExecutor(max_workers=10) as executor:
        while to_visit and n < max_pages:
            with lock:
                url = to_visit.pop(0)
            if url not in visited:
                executor.submit(visit_url, url, n)
            if not to_visit:
                time.sleep(1)
            n += 1

    urls = list(visited)
    adjacency_matrix = np.zeros((len(urls), len(urls)))
    for i, url1 in enumerate(urls):
        for j, url2 in enumerate(urls):
            if url2 in links[url1]:
                adjacency_matrix[i, j] = 1
    return adjacency_matrix, urls


def power_method(A, num_simulations: int):
    n, d = A.shape

    x = np.random.rand(n)
    x = x / np.linalg.norm(x)  # Normalize x

    for _ in range(num_simulations):
        x = np.dot(A, x)
        x = x / np.linalg.norm(x)  # Normalize x

    return x


def create_google_matrix(adjacency_matrix, alpha=0.85):
    n = adjacency_matrix.shape[0]
    S = np.divide(adjacency_matrix, adjacency_matrix.sum(axis=0))
    S[np.isnan(S)] = 1 / n
    G = alpha * S + (1 - alpha) / n * np.ones((n, n))
    return G


def get_adj(load: bool):
    if load:
        adjacency_matrix = np.load('adjacency_matrix.npy')
        with open('urls.pkl', 'rb') as f:
            urls = pickle.load(f)
            return adjacency_matrix, urls
    else:
        adjacency_matrix, urls = crawl('https://www.mit.edu/', 500)
        np.save('adjacency_matrix.npy', adjacency_matrix)
        with open('urls.pkl', 'wb') as f:
            pickle.dump(urls, f)
        return adjacency_matrix, urls


def main():
    adjacency_matrix, urls = get_adj(True)

    google_matrix = create_google_matrix(adjacency_matrix)
    dominant_eigenvector = power_method(google_matrix, num_simulations=1000)
    # Get the indices of the top 20 pages
    top_pages_indices = np.argsort(dominant_eigenvector)[-20:][::-1]

    # Print the URLs of the top 20 pages
    for i in top_pages_indices:
        print(f'URL: {urls[i]}, PageRank: {dominant_eigenvector[i]}')


if __name__ == '__main__':
    main()
