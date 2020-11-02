+++
title = "Scraping Customer Complaints - Better Business Bureau"
author = ["Sunny Hasija"]
date = 2020-11-01
draft = false
tags = ""
subtitle = "The why and how behind scraping customer complaints from the Better Business Bureau"
featured = true
categories = ""
highlight = true
+++

## Introduction {#introduction}

Customer complaints offer a unique insight into the sentiment around a product or service. From a company's perspective, scraping the reviews of their products sold on a third party platform may allow for a better understanding of the choices and preferences of their customers. Moreover, tracking customer complaints for competitors may also allow an organization to develop a competitive advantage with their own products and services.

From an academic perspective, customer complaints offer a window towards Electronic Logistics Service Quality (eLSQ) ([Rao et al. 2011](https://onlinelibrary-wiley-com.proxy.lib.ohio-state.edu/doi/full/10.1111/j.2158-1592.2011.01014.x?casa%5Ftoken=KCFnZ%5FoaccQAAAAA%3AxxzI2rZd9MEt5ZV9EN0NGUx6bLGpjFcKMuGL92FMqyxCilUoJRwBs4bApCrJynpTFuL3MmH70idNl90)). Content analysis of the complaints can help not only in qualitative studies, but could also aid in operationalizing of constructs related to eLSQ and other customer-centric theories.

This post goes over a quick method to collect data from the Better Business Bureau. This post is not meant to be comprehensive, but merely serves as a proof of concept of how such data could be collected.


## Scraping the Better Business Bureau: {#scraping-the-better-business-bureau}

For the purposes of this exercise, we will be using `Python 3.7` to scrape the data. We will make particular use of the `Selenium` libraries, and the `Firefox webdriver` to emulate the behavior of an actual user browsing the website.

Note: You can find tutorials about how to use [selenium here](https://selenium-python.readthedocs.io/). You will need to make sure you install the correct version of the webdriver based on the version of firefox you are currently using.


### Importing the libraries {#importing-the-libraries}

```python
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait                 # to wait for an element to become available on the page
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.touch_actions import TouchActions        # to allow for clicking of buttons
from selenium.webdriver.common.action_chains import ActionChains        # to allow for clicking of buttons, and linking it to other behavior
import time
```

We import the selenium and webdriver libraries, and especially the `TouchActions` and `ActionChains`, as it allows for scrolling the page, and clicking on the `next/more` buttons on the website.

The next step is to initialize the webdriver, and fetch a webpage. In this case, we are going to get the complaints from Amazon. This is especially interesting as Amazon employs workers to address the issues raised on BBB.


### Loading the webpage {#loading-the-webpage}

```python
driver=webdriver.Firefox()
source=0
driver.get("https://www.bbb.org/us/wa/seattle/profile/ecommerce/amazoncom-1296-7039385/complaints")
time.sleep(5)
```

The above code will open the complaints page for Amazon's profile on the BBB. the `time.sleep(5)` makes the script wait for 5 seconds before continuing. This ensures that the webpage is fetched and loaded before we continue.


### Gathering the data {#gathering-the-data}

There are multiple ways to get the data from this page. I am particularly interested in `Problems with the Product or Service` category of complaints. Although there exist multiple ways to navigate to this page using python, the easy way for me to scrape this category is simply click on the 11,314 (which is the number of current issues on BBB in this category) and then scrape that.

After I click on the `11,314` (as of Nov 1, 2020), the new page loads and shows me about 20 complaints at a time. At the bottom of this page is a `More` button, which upon clicking loads an additional 20 complaints, with the `More` button available again. So to load all the complaints we can iteratively scroll to the bottom of the page, click the button, and scroll to the bottom again till no more complaints are available.

After all the complaints are available, we can take all the text present on the page and dump it into a text file for further processing.
The code block below scrapes all the complaints and puts them in a text file. It also handles `TimeoutError` in case one arises.

```python
try:
    link=WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.LINK_TEXT,"11,314")))  # text for "Problem with a Product or Service"
    link.click()
    time.sleep(10)
    count=0
    while (count<400):
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            #find the more button
            more_button = WebDriverWait(driver,10).until(EC.visibility_of_element_located((By.XPATH,"/html/body/div[1]/div/div/div/main/div/div[5]/div/div[3]/button")))
            ActionChains(driver).move_to_element(more_button).click().perform()
            time.sleep(6)
            count=count+1
            print(count)
        except TimeoutError:
            break
    content= driver.find_element_by_tag_name('body')
    foo=content.text
    text_file=open("/path/to/file.txt","w")
    text_file.write(foo)
finally:
    print("Done!")
    time.sleep(3)
    driver.quit()
print("Finished!")
```


### Next Steps {#next-steps}

After the data has been written to a text file, we can do various types of text analytics on it. These might include frequency counts, sentiment analysis or topic modeling. But that is for a future blog post.
