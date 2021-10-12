const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');

/** @type {import('@docusaurus/types').DocusaurusConfig} */
module.exports = {
  title: 'Arcus - Azure Machine Learning',
  url: 'https://azureml.arcus-azure.net',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.png',
  organizationName: 'arcus-azure', // Usually your GitHub org/user name.
  projectName: 'Arcus - Azure Machine Learning', // Usually your repo name.
  themeConfig: {
    algolia: {
      apiKey: process.env.ALGOLIA_API_KEY,
      indexName: 'arcus-azure',
      searchParameters: {
        facetFilters: ["tags:azure-ml"]
      },
    },
    image: 'img/arcus.jpg',
    navbar: {
      title: 'Azure Machine Learning',
      logo: {
        alt: 'Arcus',
        src: 'img/arcus.png',
        srcDark: 'img/arcus.png'
      },
      items: [
        // Uncomment when having multiple versions
        // {
        //   type: 'docsVersionDropdown',
        //
        //   //// Optional
        //   position: 'right',
        //   // Add additional dropdown items at the beginning/end of the dropdown.
        //   dropdownItemsBefore: [],
        //   // Do not add the link active class when browsing docs.
        //   dropdownActiveClassDisabled: true,
        //   docsPluginId: 'default',
        // },
        {
          type: 'search',
          position: 'right',
        },
        {
          href: 'https://github.com/arcus-azure/arcus.azureml',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Community',
          items: [
            {
              label: 'Arcus Azure Github',
              href: 'https://github.com/arcus-azure',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()}, Arcus - Azure Machine Learning maintained by arcus-azure`,
    },
    prism: {
      theme: lightCodeTheme,
      darkTheme: darkCodeTheme,
      additionalLanguages: ['csharp'],
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          routeBasePath: "/",
          path: 'preview',
          sidebarCollapsible: false,
          // Please change this to your repo.
          editUrl:
            'https://github.com/arcus-azure/arcus.azureml/edit/master',
          // includeCurrentVersion:process.env.CONTEXT !== 'production',

        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
