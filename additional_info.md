# Additional information

## 1. Experimental Report on Differences in Information Criteria

### Table 1: Agreement between AIC, BIC, and HQIC in lag-order selection (number of groups where all criteria agree vs. not).

| | **all_equal** | **not_all_equal** |
|:---|:---:|:---:|
| **Set 1** | 5252 | 0 |
| **Set 2** | 4604 | 648 |
| **Set 3** | 5216 | 0 |
| **Set 4** | 5208 | 8 |
| **Set 5** | 5244 | 8 |
| **Set 6** | 5234 | 18 |

In Table 1, "all_equal" represents the count of cases where AIC, BIC, and HQIC completely agreed, while "not_all_equal" represents the count of cases where at least one criterion disagreed. (The total counts differ across Sets because some VAR models could not be estimated due to technical reasons.) From these results, Set 2 showed 648 cases of disagreement, but no significant differences were observed in the other Sets. A possible reason for this is that the lag is selected using Pareto optimality to ensure that the error term becomes white noise, which may minimize the lag discrepancies caused by changes in IC. (In general, the lag order typically follows AIC > HQIC > BIC.)

Total of all_equal: $5252 + 4604 + 5216 + 5208 + 5244 + 5234 = 30,758$, Total of not_all_equal: $648 + 8 + 8 + 18 = 682$, therefore $30,758/(30,758 + 682) \fallingdotseq 0.9783$, indicating that approximately $98\%$ of lags completely agreed.

Note: The actual experimental results are saved in `aic`, `bic`, and `hqic` under `var_estimation_results`. 

### Lag Distribution for Each IC in Each Set

<img src="https://anonymous.4open.science/api/repo/982026118ecd19e44cd1db12243eebd3/file/resources/lags-distribution.png?v=5c185dce">

### Decision Tree Classification Results

The results for AIC, BIC, and HQIC were all identical. This demonstrates the robustness of MIAO.

#### AIC

| Metric           | Set 1 | Set 2 | Set 3 | Set 4 | Set 5 | Set 6 | Mean ± SD   | N     |
|------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:-----:|
| Accuracy         | 0.81  | 0.79  | 0.67  | 0.74  | 0.74  | 0.77  | 0.75 ± 0.05 | 187.0 |
| REV F1-Score     | 0.79  | 0.78  | 0.57  | 0.70  | 0.72  | 0.72  | 0.71 ± 0.07 | 87.0  |
| non-REV F1-Score | 0.83  | 0.80  | 0.73  | 0.77  | 0.76  | 0.81  | 0.78 ± 0.03 | 100.0 |

#### BIC

| Metric           | Set 1 | Set 2 | Set 3 | Set 4 | Set 5 | Set 6 | Mean ± SD   | N     |
|------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:-----:|
| Accuracy         | 0.81  | 0.79  | 0.67  | 0.74  | 0.74  | 0.77  | 0.75 ± 0.05 | 187.0 |
| REV F1-Score     | 0.79  | 0.78  | 0.57  | 0.70  | 0.72  | 0.72  | 0.71 ± 0.07 | 87.0  |
| non-REV F1-Score | 0.83  | 0.80  | 0.73  | 0.77  | 0.76  | 0.81  | 0.78 ± 0.03 | 100.0 |

#### HQIC

| Metric           | Set 1 | Set 2 | Set 3 | Set 4 | Set 5 | Set 6 | Mean ± SD   | N     |
|------------------|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----------:|:-----:|
| Accuracy         | 0.81  | 0.79  | 0.67  | 0.74  | 0.74  | 0.77  | 0.75 ± 0.05 | 187.0 |
| REV F1-Score     | 0.79  | 0.78  | 0.57  | 0.70  | 0.72  | 0.72  | 0.71 ± 0.07 | 87.0  |
| non-REV F1-Score | 0.83  | 0.80  | 0.73  | 0.77  | 0.76  | 0.81  | 0.78 ± 0.03 | 100.0 |

# Supplementary Information on the Original REV Candidates

## Table 2: Summary of Exclusion Reasons for 147 Projects Excluded from the REV Group

Below is a table listing the projects excluded from the REV group. These 147 projects, together with the [87 projects](https://anonymous.4open.science/r/982026118ecd19e44cd1db12243eebd3/final_dataset.csv) used in the actual experiment, constitute the original 234 REV candidates.

### Column Descriptions

* `Repository`: The name of the excluded repository.
* `Excluded reason`: The reason for exclusion, categorized into the following 7 types:
    * UNIDENTIFIED_COMPETITOR: Difficult to identify competitors
    * PROJECT_MIGRATION: Transferred to another repository
    * OFFICIAL_NATIVE_REPLACEMENT: Replaced by native or official implementation
    * NON_OSS_COMPETITOR: Competitor is not open source
    * INSUFFICIENT_OVERLAP: Overlap period with competitor is less than 1 year
    * DATA_DISCREPANCY: Discrepancy with GHS data (commit < 500, alive project, etc.)
    * SELF_FORK_COMPETITION: Competitor includes the project's own fork
* `Note`: Notes from the experiment including migration destinations or reasons for exclusion

**Note: While the table contains 149 records, the actual N is 147 because `mycroftai/mycroft-core` and `nylas/nylas-mail` each have two fork destinations.**


|     | Repository                                   | Excluded reason             | Note                                                                         |
|----:|:---------------------------------------------|:----------------------------|:-----------------------------------------------------------------------------|
|   0 | a8m/angular-filter                           | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than a8m/ng-pipes                    |
|   1 | adw0rd/instagrapi                            | INSUFFICIENT_OVERLAP        | nan                                                                          |
|   2 | afnetworking/afnetworking                    | INSUFFICIENT_OVERLAP        | nan                                                                          |
|   3 | ageron/handson-ml2                           | PROJECT_MIGRATION           | ageron/handson-ml3                                                           |
|   4 | alecaivazis/survey                           | INSUFFICIENT_OVERLAP        | nan                                                                          |
|   5 | amirzaidi/launcher3                          | SELF_FORK_COMPETITION       | LawnchairLauncher/lawnchair                                                  |
|   6 | anbox/anbox                                  | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than waydroid/waydroid               |
|   7 | apprenticeharper/dedrm_tools                 | SELF_FORK_COMPETITION       | nodrm/DeDRM_tools                                                            |
|   8 | appsquickly/typhoon                          | INSUFFICIENT_OVERLAP        | nan                                                                          |
|   9 | ar-js-org/ar.js                              | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than hiukim/mind-ar-js               |
|  10 | arthurhub/android-image-cropper              | SELF_FORK_COMPETITION       | CanHub/Android-Image-Cropper                                                 |
|  11 | attic-labs/noms                              | SELF_FORK_COMPETITION       | dolthub/dolt                                                                 |
|  12 | austintgriffith/scaffold-eth                 | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than austintgriffith/scaffold-eth    |
|  13 | axnsan12/drf-yasg                            | DATA_DISCREPANCY            | nan                                                                          |
|  14 | beyondcode/laravel-websockets                | OFFICIAL_NATIVE_REPLACEMENT | laravel/reverb                                                               |
|  15 | bloodhoundad/bloodhound                      | PROJECT_MIGRATION           | SpecterOps/BloodHound                                                        |
|  16 | boltdb/bolt                                  | SELF_FORK_COMPETITION       | coreos/bbolt                                                                 |
|  17 | boto/boto                                    | OFFICIAL_NATIVE_REPLACEMENT | Migrated to boto/boto3                                                       |
|  18 | bvaughn/react-virtualized                    | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than bvaughn/react-window            |
|  19 | callstack/haul                               | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  20 | claracrazy/flipper-xtreme                    | SELF_FORK_COMPETITION       | DarkFlippers/unleashed-firmware                                              |
|  21 | codemirror/codemirror                        | DATA_DISCREPANCY            | nan                                                                          |
|  22 | containrrr/watchtower                        | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  23 | davezuko/react-redux-starter-kit             | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  24 | defunkt/jquery-pjax                          | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  25 | depcheck/depcheck                            | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than webpro-nl/knip                  |
|  26 | didierrlopes/gamestonkterminal               | PROJECT_MIGRATION           | OpenBB-finance/OpenBBTerminal                                                |
|  27 | dingo/api                                    | SELF_FORK_COMPETITION       | api-ecosystem-for-laravel/dingo-api                                          |
|  28 | docker-archive/docker-registry               | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  29 | domaindrivendev/swashbuckle.webapi           | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  30 | dotnet/corert                                | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  31 | dresende/node-orm2                           | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  32 | dropbox/lepton                               | UNIDENTIFIED_COMPETITOR     | Successors are WebP and JPEG XL, but they are specifications                 |
|  33 | dtan4/terraforming                           | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  34 | eliaskotlyar/xiaomi-dafang-hacks             | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  35 | encode/apistar                               | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than Kludex/starlette                |
|  36 | erming/shout                                 | SELF_FORK_COMPETITION       | thelounge/thelounge                                                          |
|  37 | exyte/macaw                                  | OFFICIAL_NATIVE_REPLACEMENT | https://developer.apple.com/swiftui/                                         |
|  38 | facebook/draft-js                            | PROJECT_MIGRATION           | facebook/lexical                                                             |
|  39 | facebookarchive/draft-js                     | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  40 | facebookarchive/flux                         | DATA_DISCREPANCY            | nan                                                                          |
|  41 | facebookarchive/webdriveragent               | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than facebook/idb                    |
|  42 | fdehau/tui-rs                                | SELF_FORK_COMPETITION       | ratatui-org/ratatui                                                          |
|  43 | fent/node-ytdl-core                          | SELF_FORK_COMPETITION       | distubejs/ytdl-core                                                          |
|  44 | flif-hub/flif                                | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than google/pik                      |
|  45 | flutter/gallery                              | UNIDENTIFIED_COMPETITOR     | nan                                                                          |
|  46 | foreversd/forever                            | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  47 | gaearon/react-hot-loader                     | OFFICIAL_NATIVE_REPLACEMENT | https://github.com/facebook/react/issues/16604                               |
|  48 | ganapati/rsactftool                          | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  49 | geekyants/nativebase                         | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than https://gluestack.io/           |
|  50 | github-for-unity/unity                       | SELF_FORK_COMPETITION       | spoiledcat/git-for-unity                                                     |
|  51 | go-survey/survey                             | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than charmbracelet/bubbletea         |
|  52 | golang/dep                                   | PROJECT_MIGRATION           | Integrated into golang/go                                                    |
|  53 | google/iosched                               | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  54 | google/netstack                              | PROJECT_MIGRATION           | google/gvisor                                                                |
|  55 | googlecontainertools/container-diff          | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than reproducible-containers/diffoci |
|  56 | iissnan/hexo-theme-next                      | PROJECT_MIGRATION           | theme-next/hexo-theme-next                                                   |
|  57 | insin/nwb                                    | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than vitejs/vite                     |
|  58 | ionic-team/ng-cordova                        | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  59 | ipfs/js-ipfs                                 | PROJECT_MIGRATION           | ipfs/helia                                                                   |
|  60 | jakewharton/actionbarsherlock                | OFFICIAL_NATIVE_REPLACEMENT | https://developer.android.com/develop/ui/views/components/appbar             |
|  61 | josephg/sharejs                              | PROJECT_MIGRATION           | share/sharedb                                                                |
|  62 | jp9000/obs                                   | PROJECT_MIGRATION           | obsproject/obs-studio                                                        |
|  63 | justadudewhohacks/opencv4nodejs              | SELF_FORK_COMPETITION       | UrielCh/opencv4nodejs                                                        |
|  64 | jwyang/faster-rcnn.pytorch                   | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  65 | kazupon/vue-i18n                             | SELF_FORK_COMPETITION       | intlify/vue-i18n-next                                                        |
|  66 | linuxbrew/brew                               | SELF_FORK_COMPETITION       | Homebrew/brew                                                                |
|  67 | lipangit/jiaozivideoplayer                   | SELF_FORK_COMPETITION       | Jzvd/JiaoZiVideoPlayer                                                       |
|  68 | luin/ioredis                                 | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than redis/node-redis                |
|  69 | mac-s-g/react-json-view                      | SELF_FORK_COMPETITION       | microlinkhq/react-json-view                                                  |
|  70 | marmelab/ng-admin                            | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  71 | mcxiaoke/android-volley                      | SELF_FORK_COMPETITION       | google/volley                                                                |
|  72 | mengshukeji/luckysheet                       | PROJECT_MIGRATION           | dream-num/univer                                                             |
|  73 | microsoft/git-credential-manager-for-windows | OFFICIAL_NATIVE_REPLACEMENT | microsoft/Git-Credential-Manager-Core                                        |
|  74 | mislav/hub                                   | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than cli/cli                         |
|  75 | mycroftai/mycroft-core                       | SELF_FORK_COMPETITION       | OpenVoiceOS/ovos-core                                                        |
|  76 | mycroftai/mycroft-core                       | SELF_FORK_COMPETITION       | NeonGeckoCom/NeonCore                                                        |
|  77 | mysqljs/mysql                                | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  78 | naman14/timber                               | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  79 | nanomsg/nanomsg                              | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  80 | netflix/hystrix                              | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  81 | neuecc/unirx                                 | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than Cysharp/R3                      |
|  82 | nicoespeon/gitgraph.js                       | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  83 | nsf/gocode                                   | OFFICIAL_NATIVE_REPLACEMENT | https://pkg.go.dev/golang.org/x/tools/gopls                                  |
|  84 | nylas/nylas-mail                             | SELF_FORK_COMPETITION       | nylas-mail-lives/nylas-mail                                                  |
|  85 | nylas/nylas-mail                             | SELF_FORK_COMPETITION       | Foundry376/Mailspring                                                        |
|  86 | ohmyform/ohmyform                            | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than formbricks/formbricks           |
|  87 | open-falcon/falcon-plus                      | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  88 | openai/gym                                   | SELF_FORK_COMPETITION       | Farama-Foundation/Gymnasium                                                  |
|  89 | openkinect/libfreenect                       | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  90 | openstf/stf                                  | SELF_FORK_COMPETITION       | DeviceFarmer/stf                                                             |
|  91 | owlcarousel2/owlcarousel2                    | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  92 | panqiwei/autogptq                            | SELF_FORK_COMPETITION       | ModelCloud/GPTQModel                                                         |
|  93 | phpoffice/phpexcel                           | SELF_FORK_COMPETITION       | PHPOffice/PhpSpreadsheet                                                     |
|  94 | picocms/pico                                 | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  95 | polymorphicshade/newpipe                     | SELF_FORK_COMPETITION       | polymorphicshade/Tubular                                                     |
|  96 | powerlevel9k/powerlevel9k                    | PROJECT_MIGRATION           | romkatv/powerlevel10k                                                        |
|  97 | prisma-archive/chromeless                    | INSUFFICIENT_OVERLAP        | nan                                                                          |
|  98 | pyenv/pyenv                                  | UNIDENTIFIED_COMPETITOR     | nan                                                                          |
|  99 | qtox/qtox                                    | SELF_FORK_COMPETITION       | TokTok/qTox                                                                  |
| 100 | r-darwish/topgrade                           | SELF_FORK_COMPETITION       | topgrade-rs/topgrade                                                         |
| 101 | radiant-player/radiant-player-mac            | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors                                            |
| 102 | rancher/k3os                                 | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 103 | raynos/mercury                               | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than optoolco/tonic                  |
| 104 | rcore-os/rcore                               | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 105 | react-native-camera/react-native-camera      | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 106 | react-static/react-static                    | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 107 | redfin/react-server                          | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 108 | rustformers/llm                              | INSUFFICIENT_OVERLAP        | Overlap period is less than 1 year                                           |
| 109 | searx/searx                                  | SELF_FORK_COMPETITION       | searxng/searxng                                                              |
| 110 | serverless-nextjs/serverless-next.js         | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 111 | sferik/twitter                               | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than sferik/x-ruby                   |
| 112 | sightmachine/simplecv                        | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 113 | skoruba/identityserver4.admin                | SELF_FORK_COMPETITION       | skoruba/Duende.IdentityServer.Admin                                          |
| 114 | solnic/virtus                                | UNIDENTIFIED_COMPETITOR     | nan                                                                          |
| 115 | spacecloud-io/space-cloud                    | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 116 | spencerwooo/onedrive-vercel-index            | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than alist-org/alist                 |
| 117 | spikecodes/libreddit                         | PROJECT_MIGRATION           | redlib-org/redlib                                                            |
| 118 | spotify/docker-maven-plugin                  | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than spotify/dockerfile-maven        |
| 119 | square/dagger                                | SELF_FORK_COMPETITION       | google/dagger                                                                |
| 120 | squizlabs/php_codesniffer                    | SELF_FORK_COMPETITION       | PHPCSStandards/PHP_CodeSniffer                                               |
| 121 | stackexchange/blackbox                       | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 122 | steipete/pstcollectionview                   | OFFICIAL_NATIVE_REPLACEMENT | 100% API compatible replacement of UICollectionView for iOS4.3+              |
| 123 | streadway/amqp                               | SELF_FORK_COMPETITION       | rabbitmq/amqp091-go                                                          |
| 124 | swiftmailer/swiftmailer                      | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 125 | swiip/generator-gulp-angular                 | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 126 | swooletw/laravel-swoole                      | OFFICIAL_NATIVE_REPLACEMENT | laravel/octane                                                               |
| 127 | teampoltergeist/poltergeist                  | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 128 | tellform/tellform                            | SELF_FORK_COMPETITION       | ohmyform/ohmyform                                                            |
| 129 | the-paperless-project/paperless              | SELF_FORK_COMPETITION       | jonaswinkler/paperless-ng                                                    |
| 130 | theano/theano                                | SELF_FORK_COMPETITION       | pymc-devs/pytensor                                                           |
| 131 | thestinger/termite                           | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 132 | trentrichardson/jquery-timepicker-addon      | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than flatpickr/flatpickr             |
| 133 | videojs/videojs-contrib-hls                  | SELF_FORK_COMPETITION       | videojs/http-streaming                                                       |
| 134 | volatilityfoundation/volatility              | PROJECT_MIGRATION           | volatilityfoundation/volatility3                                             |
| 135 | vuejs-templates/webpack                      | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 136 | vuejs/vetur                                  | PROJECT_MIGRATION           | vuejs/language-tools                                                         |
| 137 | vuejs/vuex                                   | OFFICIAL_NATIVE_REPLACEMENT | vuejs/pinia                                                                  |
| 138 | vurtun/nuklear                               | SELF_FORK_COMPETITION       | Immediate-Mode-UI/Nuklear                                                    |
| 139 | w3c/intersectionobserver                     | PROJECT_MIGRATION           | GoogleChromeLabs/intersection-observer                                       |
| 140 | wal-e/wal-e                                  | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 141 | wbthomason/packer.nvim                       | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 142 | xmartlabs/xlform                             | UNIDENTIFIED_COMPETITOR     | Difficult to identify competitors other than xmartlabs/Eureka                |
| 143 | yelp/elastalert                              | SELF_FORK_COMPETITION       | jertel/elastalert2                                                           |
| 144 | zackchase/mxnet-the-straight-dope            | PROJECT_MIGRATION           | https://d2l.ai/                                                              |
| 145 | zalmoxisus/redux-devtools-extension          | PROJECT_MIGRATION           | reduxjs/redux-devtools                                                       |
| 146 | zeroclipboard/zeroclipboard                  | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 147 | zhukov/webogram                              | INSUFFICIENT_OVERLAP        | nan                                                                          |
| 148 | zo0r/react-native-push-notification          | INSUFFICIENT_OVERLAP        | nan                                                                          |

## Breakdown of Exclusion Reasons

<img src="https://anonymous.4open.science/api/repo/982026118ecd19e44cd1db12243eebd3/file/resources/percentage_of_excluded_reason.png?v=5847d828">

