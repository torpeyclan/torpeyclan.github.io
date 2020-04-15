---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

<div class="container">
<h1 class="display-2 font-weight-bold mt-5"><span style="color:#2045d7">Torpey Clan</span></h1>

<p class="lead font-weight-normal">
<span>
Torpey Clan Blog
</span>
</p>
</div>

<div style="background-color:#ffffff" class="jumbotron jumbotron-fluid">
    <div style="background-color:#ffffff" class="container">
        <div class="row">
            <div class="col-sm-12">
                <a style="color:#000000" class="display-4 font-weight-bold" href="blog">
                    Posts.
                </a>
                <p class="lead font-weight-bold">
                <span></span>
                </p>
            </div>
        </div>
        <div class="card-columns">
        {% for post in site.posts %}
         <article>
                <div style="margin-bottom:16px;padding:24px" class="card">
                    <h2>
                        <a style="color:#000000" href="{{ post.url }}">
                            {{ post.title }}
                        </a>
                    </h2>
                    <time class="time" datetime="{{ post.date | date: "%Y-%m-%d" }}">{{ post.date | date_to_long_string }}</time>
                    <p class="text">
                        {{post.excerpt }}
                    </p>
                    {% if post.tags.size > 0 %}
                    <p>
                        <span>
                         {% for tag in post.tags %}
                            <span style="color:#ffffff;background-color:#000000" class="badge badge-secondary">
                               {{tag}}
                            </span>
                            &nbsp;
                        {% endfor %}
                        </span>
                    </p>
                    {% endif %}
                </div>
            </article>
            {% endfor %}
        </div>
    </div>
</div>
