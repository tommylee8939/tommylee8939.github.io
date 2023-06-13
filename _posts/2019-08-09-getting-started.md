---
title: Getting Started
author: cotes
date: 2019-08-09 20:55:00 +0800
categories: [Blogging, Tutorial]
tags: [getting started]
pin: true
---

아래의 설명은 Chirpy 원본 document에서 차용하였다.



## Jekyll
가장 우선적으로 github에 repo를 생성하기 전에 jekyll 를 사용할 수 있는 환경을 조성하자. (Mac os 기준으로 설명)

1. Homebrew 통해서 rbenv, ruby-build 설치
   <br>
   ```console
   $ brew install rbenv ruby-build
   ``` 
   
    ruby, ruby-build에 대해서는 자세하게 알 필요는 없다. 필자도 정확하게 알지는 못한다. 다만 rbenv 통해서 여러 종류의 Ruby를 설치할 수 있게 지원한다. ruby-build라는 플러그인을 통해 설치 가능하고 이를 통해 다른 버전의 Ruby들을 임의의 디렉토리에 컴파일하고 설치할 수 있게 도와준다.
2. ruby 설치
    ```console
    $ rbenv install -l    # 설치 할수 있는 버전 확인

    $ rbenv install 2.7.5 # 설치

    $ rbenv global 2.7.5 

    $ rbenv rehash 

    (base) ➜  ~ rbenv versions
      system
    * 2.7.5 (set by /Users/iganghui/.rbenv/version) # 최종적으로 설치한 버전에 *가 오면 된다
    ```
3. rbenv path 추가 (설정 하지 않으면 4번에서 에러 발생한다.)

   시스템 상에도 Ruby가 존재하는데 방금 설치한 Ruby 버전에 접근할 수 있도록 수정한다.
    ```console
    $ open ~/.zshrc
    ```
    아래 두줄 을 추가해준다.
    ```
    export PATH={$Home}/.rbenv/bin:$PATH && \
    eval "$(rbenv init -)"
    ```
4. gem 패키지 설치
    
   ```console
   $ gem install bundler
   $ gem install jekyll
   ```

> 필요한 패키지들은 다 설치하였다. 
> (혹시나 설치하다가 문제가 발생하였을 경우 다음의 파일들을 다 삭제해주면 된다.)

```console
$ gem uninstall -aIx  # gem packages 다 지운다.
$ rm -rf ~/.bundle
$ rm -rf ~/.gem
$ rm -rf ~/.rbenv
$ brew uninstall rbenv ruby-build
```


## Installation

### Creating a New Site

There are two ways to create a new repository for this theme:

- [**Using the Chirpy Starter**](#option-1-using-the-chirpy-starter) - Easy to upgrade, isolates irrelevant project files so you can focus on writing.
- [**GitHub Fork**](#option-2-github-fork) - Convenient for custom development, but difficult to upgrade. Unless you are familiar with Jekyll and are determined to tweak or contribute to this project, this approach is not recommended.

#### Option 1. Using the Chirpy Starter

Sign in to GitHub and browse to [**Chirpy Starter**][starter], click the button <kbd>Use this template</kbd> > <kbd>Create a new repository</kbd>, and name the new repository `USERNAME.github.io`, where `USERNAME` represents your GitHub username.

#### Option 2. GitHub Fork

Sign in to GitHub to [fork **Chirpy**](https://github.com/cotes2020/jekyll-theme-chirpy/fork), and then rename it to `USERNAME.github.io` (`USERNAME` means your username).

Next, clone your site to local machine. In order to build JavaScript files later, we need to install [Node.js][nodejs], and then run the tool:

```console
$ bash tools/init
```

> If you don't want to deploy your site on GitHub Pages, append option `--no-gh` at the end of the above command.
{: .prompt-info }

The above command will:

1. Check out the code to the [latest tag][latest-tag] (to ensure the stability of your site: as the code for the default branch is under development).
2. Remove non-essential sample files and take care of GitHub-related files.
3. Build JavaScript files and export to `assets/js/dist/`{: .filepath }, then make them tracked by Git.
4. Automatically create a new commit to save the changes above.

### Installing Dependencies

Before running local server for the first time, go to the root directory of your site and run:

```console
$ bundle
```

## Usage

### Configuration

Update the variables of `_config.yml`{: .filepath} as needed. Some of them are typical options:

- `url`
- `avatar`
- `timezone`
- `lang`

### Customizing Stylesheet

If you need to customize the stylesheet, copy the theme's `assets/css/style.scss`{: .filepath} to the same path on your Jekyll site, and then add the custom style at the end of it.

Starting with version `4.1.0`, if you want to overwrite the SASS variables defined in `_sass/addon/variables.scss`{: .filepath}, copy the main sass file `_sass/jekyll-theme-chirpy.scss`{: .filepath} into the `_sass`{: .filepath} directory in your site's source, then create a new file `_sass/variables-hook.scss`{: .filepath} and assign new value.

### Customing Static Assets

Static assets configuration was introduced in version `5.1.0`. The CDN of the static assets is defined by file `_data/origin/cors.yml`{: .filepath }, and you can replace some of them according to the network conditions in the region where your website is published.

Also, if you'd like to self-host the static assets, please refer to the [_chirpy-static-assets_](https://github.com/cotes2020/chirpy-static-assets#readme).

### Running Local Server

You may want to preview the site contents before publishing, so just run it by:

```console
$ bundle exec jekyll s
```

Or run the site on Docker with the following command:

```console
$ docker run -it --rm \
    --volume="$PWD:/srv/jekyll" \
    -p 4000:4000 jekyll/jekyll \
    jekyll serve
```

After a few seconds, the local service will be published at _<http://127.0.0.1:4000>_.

## Deployment

Before the deployment begins, check out the file `_config.yml`{: .filepath} and make sure the `url` is configured correctly. Furthermore, if you prefer the [**project site**](https://help.github.com/en/github/working-with-github-pages/about-github-pages#types-of-github-pages-sites) and don't use a custom domain, or you want to visit your website with a base URL on a web server other than **GitHub Pages**, remember to change the `baseurl` to your project name that starts with a slash, e.g, `/project-name`.

Now you can choose _ONE_ of the following methods to deploy your Jekyll site.

### Deploy by Using GitHub Actions

There are a few things to get ready for.

- If you're on the GitHub Free plan, keep your site repository public.
- If you have committed `Gemfile.lock`{: .filepath} to the repository, and your local machine is not running Linux, go the the root of your site and update the platform list of the lock-file:

  ```console
  $ bundle lock --add-platform x86_64-linux
  ```

Next, configure the _Pages_ service.

1. Browse to your repository on GitHub. Select the tab _Settings_, then click _Pages_ in the left navigation bar. Then, in the **Source** section (under _Build and deployment_), select [**GitHub Actions**][pages-workflow-src] from the dropdown menu.

2. Push any commits to GitHub to trigger the _Actions_ workflow. In the _Actions_ tab of your repository, you should see the workflow _Build and Deploy_ running. Once the build is complete and successful, the site will be deployed automatically.

At this point, you can go to the URL indicated by GitHub to access your site.

### Manually Build and Deploy

On self-hosted servers, you cannot enjoy the convenience of **GitHub Actions**. Therefore, you should build the site on your local machine and then upload the site files to the server.

Go to the root of the source project, and build your site as follows:

```console
$ JEKYLL_ENV=production bundle exec jekyll b
```

Or build the site on Docker:

```console
$ docker run -it --rm \
    --env JEKYLL_ENV=production \
    --volume="$PWD:/srv/jekyll" \
    jekyll/jekyll \
    jekyll build
```

Unless you specified the output path, the generated site files will be placed in folder `_site`{: .filepath} of the project's root directory. Now you should upload those files to the target server.

[nodejs]: https://nodejs.org/
[starter]: https://github.com/cotes2020/chirpy-starter
[pages-workflow-src]: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site#publishing-with-a-custom-github-actions-workflow
[latest-tag]: https://github.com/cotes2020/jekyll-theme-chirpy/tags
